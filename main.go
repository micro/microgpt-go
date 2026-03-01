// The most atomic way to train and run inference for a GPT in pure, dependency-free Go.
// This file is the complete algorithm.
// Everything else is just efficiency.
//
// Translated from the Python original by @karpathy:
// https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
package main

import (
	"bufio"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net/http"
	"os"
	"sort"
	"strings"
)

// --- Autograd Engine ---

// Value represents a node in the computation graph for automatic differentiation.
type Value struct {
	data       float64
	grad       float64
	children   []*Value
	localGrads []float64
}

// NewValue creates a leaf node with the given scalar value.
func NewValue(data float64) *Value {
	return &Value{data: data}
}

// Add returns a new Value representing a + b.
func (a *Value) Add(b *Value) *Value {
	return &Value{
		data:       a.data + b.data,
		children:   []*Value{a, b},
		localGrads: []float64{1, 1},
	}
}

// AddScalar returns a new Value representing a + s.
func (a *Value) AddScalar(s float64) *Value {
	return a.Add(NewValue(s))
}

// Mul returns a new Value representing a * b.
func (a *Value) Mul(b *Value) *Value {
	return &Value{
		data:       a.data * b.data,
		children:   []*Value{a, b},
		localGrads: []float64{b.data, a.data},
	}
}

// MulScalar returns a new Value representing a * s.
func (a *Value) MulScalar(s float64) *Value {
	return a.Mul(NewValue(s))
}

// Pow returns a new Value representing a^n.
func (a *Value) Pow(n float64) *Value {
	return &Value{
		data:       math.Pow(a.data, n),
		children:   []*Value{a},
		localGrads: []float64{n * math.Pow(a.data, n-1)},
	}
}

// Log returns a new Value representing ln(a).
func (a *Value) Log() *Value {
	return &Value{
		data:       math.Log(a.data),
		children:   []*Value{a},
		localGrads: []float64{1.0 / a.data},
	}
}

// Exp returns a new Value representing exp(a).
func (a *Value) Exp() *Value {
	expVal := math.Exp(a.data)
	return &Value{
		data:       expVal,
		children:   []*Value{a},
		localGrads: []float64{expVal},
	}
}

// ReLU returns a new Value representing max(0, a).
func (a *Value) ReLU() *Value {
	data := 0.0
	grad := 0.0
	if a.data > 0 {
		data = a.data
		grad = 1.0
	}
	return &Value{
		data:       data,
		children:   []*Value{a},
		localGrads: []float64{grad},
	}
}

// Neg returns a new Value representing -a.
func (a *Value) Neg() *Value {
	return a.MulScalar(-1)
}

// Sub returns a new Value representing a - b.
func (a *Value) Sub(b *Value) *Value {
	return a.Add(b.Neg())
}

// Div returns a new Value representing a / b.
func (a *Value) Div(b *Value) *Value {
	return a.Mul(b.Pow(-1))
}

// Backward computes gradients for all nodes in the computation graph via backpropagation.
func (v *Value) Backward() {
	// Topological sort
	topo := make([]*Value, 0)
	visited := make(map[*Value]bool)
	var buildTopo func(*Value)
	buildTopo = func(node *Value) {
		if visited[node] {
			return
		}
		visited[node] = true
		for _, child := range node.children {
			buildTopo(child)
		}
		topo = append(topo, node)
	}
	buildTopo(v)
	v.grad = 1
	for i := len(topo) - 1; i >= 0; i-- {
		node := topo[i]
		for j, child := range node.children {
			child.grad += node.localGrads[j] * node.grad
		}
	}
}

// --- Neural Network Helper Functions ---

// linear computes a matrix-vector product: y = W * x, where W is [nout x nin] and x is [nin].
func linear(x []*Value, w [][]*Value) []*Value {
	result := make([]*Value, len(w))
	for i, wo := range w {
		sum := wo[0].Mul(x[0])
		for j := 1; j < len(wo); j++ {
			sum = sum.Add(wo[j].Mul(x[j]))
		}
		result[i] = sum
	}
	return result
}

// softmax computes the softmax of logits with numerical stability.
func softmax(logits []*Value) []*Value {
	maxVal := logits[0].data
	for _, v := range logits[1:] {
		if v.data > maxVal {
			maxVal = v.data
		}
	}
	exps := make([]*Value, len(logits))
	for i, v := range logits {
		exps[i] = v.AddScalar(-maxVal).Exp()
	}
	total := exps[0]
	for i := 1; i < len(exps); i++ {
		total = total.Add(exps[i])
	}
	result := make([]*Value, len(exps))
	for i, e := range exps {
		result[i] = e.Div(total)
	}
	return result
}

// rmsnorm computes the Root Mean Square Layer Normalization.
func rmsnorm(x []*Value) []*Value {
	n := float64(len(x))
	ms := x[0].Mul(x[0])
	for i := 1; i < len(x); i++ {
		ms = ms.Add(x[i].Mul(x[i]))
	}
	ms = ms.MulScalar(1.0 / n)
	scale := ms.AddScalar(1e-5).Pow(-0.5)
	result := make([]*Value, len(x))
	for i, xi := range x {
		result[i] = xi.Mul(scale)
	}
	return result
}

// matrix creates a [nout x nin] matrix of Values initialized from a Gaussian distribution.
func matrix(rng *rand.Rand, nout, nin int, std float64) [][]*Value {
	m := make([][]*Value, nout)
	for i := 0; i < nout; i++ {
		m[i] = make([]*Value, nin)
		for j := 0; j < nin; j++ {
			m[i][j] = NewValue(rng.NormFloat64() * std)
		}
	}
	return m
}

// weightedChoice selects an index with probability proportional to the given weights.
func weightedChoice(rng *rand.Rand, weights []float64) int {
	total := 0.0
	for _, w := range weights {
		total += w
	}
	r := rng.Float64() * total
	cumulative := 0.0
	for i, w := range weights {
		cumulative += w
		if r < cumulative {
			return i
		}
	}
	return len(weights) - 1
}

func main() {
	rng := rand.New(rand.NewSource(42)) // Let there be order among chaos

	// --- Dataset ---
	// Let there be a Dataset `docs`: list of documents (e.g. a list of names)
	if _, err := os.Stat("input.txt"); os.IsNotExist(err) {
		namesURL := "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"
		resp, err := http.Get(namesURL)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error downloading dataset: %v\n", err)
			os.Exit(1)
		}
		defer resp.Body.Close()
		out, err := os.Create("input.txt")
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error creating input.txt: %v\n", err)
			os.Exit(1)
		}
		if _, err = io.Copy(out, resp.Body); err != nil {
			out.Close()
			fmt.Fprintf(os.Stderr, "Error writing input.txt: %v\n", err)
			os.Exit(1)
		}
		out.Close()
	}

	file, err := os.Open("input.txt")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error opening input.txt: %v\n", err)
		os.Exit(1)
	}
	var docs []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line != "" {
			docs = append(docs, line)
		}
	}
	file.Close()
	rng.Shuffle(len(docs), func(i, j int) { docs[i], docs[j] = docs[j], docs[i] })
	fmt.Printf("num docs: %d\n", len(docs))

	// --- Tokenizer ---
	// Let there be a Tokenizer to translate strings to sequences of integers ("tokens") and back
	charSet := make(map[rune]bool)
	for _, doc := range docs {
		for _, ch := range doc {
			charSet[ch] = true
		}
	}
	uchars := make([]rune, 0, len(charSet))
	for ch := range charSet {
		uchars = append(uchars, ch)
	}
	sort.Slice(uchars, func(i, j int) bool { return uchars[i] < uchars[j] })

	charToIdx := make(map[rune]int)
	for i, ch := range uchars {
		charToIdx[ch] = i
	}

	BOS := len(uchars) // token id for a special Beginning of Sequence (BOS) token
	vocabSize := len(uchars) + 1
	fmt.Printf("vocab size: %d\n", vocabSize)

	// --- Model Hyperparameters ---
	nLayer := 1     // depth of the transformer neural network (number of layers)
	nEmbd := 16     // width of the network (embedding dimension)
	blockSize := 16 // maximum context length of the attention window
	nHead := 4      // number of attention heads
	headDim := nEmbd / nHead

	// --- Initialize Parameters ---
	std := 0.08
	stateDict := make(map[string][][]*Value)
	stateDict["wte"] = matrix(rng, vocabSize, nEmbd, std)
	stateDict["wpe"] = matrix(rng, blockSize, nEmbd, std)
	stateDict["lm_head"] = matrix(rng, vocabSize, nEmbd, std)
	for i := 0; i < nLayer; i++ {
		stateDict[fmt.Sprintf("layer%d.attn_wq", i)] = matrix(rng, nEmbd, nEmbd, std)
		stateDict[fmt.Sprintf("layer%d.attn_wk", i)] = matrix(rng, nEmbd, nEmbd, std)
		stateDict[fmt.Sprintf("layer%d.attn_wv", i)] = matrix(rng, nEmbd, nEmbd, std)
		stateDict[fmt.Sprintf("layer%d.attn_wo", i)] = matrix(rng, nEmbd, nEmbd, std)
		stateDict[fmt.Sprintf("layer%d.mlp_fc1", i)] = matrix(rng, 4*nEmbd, nEmbd, std)
		stateDict[fmt.Sprintf("layer%d.mlp_fc2", i)] = matrix(rng, nEmbd, 4*nEmbd, std)
	}

	// Flatten all parameters into a single list (in deterministic insertion order)
	paramKeys := []string{"wte", "wpe", "lm_head"}
	for i := 0; i < nLayer; i++ {
		paramKeys = append(paramKeys,
			fmt.Sprintf("layer%d.attn_wq", i),
			fmt.Sprintf("layer%d.attn_wk", i),
			fmt.Sprintf("layer%d.attn_wv", i),
			fmt.Sprintf("layer%d.attn_wo", i),
			fmt.Sprintf("layer%d.mlp_fc1", i),
			fmt.Sprintf("layer%d.mlp_fc2", i),
		)
	}
	var params []*Value
	for _, key := range paramKeys {
		for _, row := range stateDict[key] {
			params = append(params, row...)
		}
	}
	fmt.Printf("num params: %d\n", len(params))

	// --- GPT Model ---
	// Define the model architecture: a function mapping tokens and parameters to logits
	// Follows GPT-2 with minor differences: layernorm -> rmsnorm, no biases, GeLU -> ReLU
	gpt := func(tokenID, posID int, keys, values [][][]*Value) []*Value {
		tokEmb := stateDict["wte"][tokenID]
		posEmb := stateDict["wpe"][posID]
		x := make([]*Value, nEmbd)
		for i := 0; i < nEmbd; i++ {
			x[i] = tokEmb[i].Add(posEmb[i])
		}
		x = rmsnorm(x) // not redundant due to backward pass via the residual connection

		for li := 0; li < nLayer; li++ {
			// 1) Multi-head Attention block
			xResidual := x
			x = rmsnorm(x)
			q := linear(x, stateDict[fmt.Sprintf("layer%d.attn_wq", li)])
			k := linear(x, stateDict[fmt.Sprintf("layer%d.attn_wk", li)])
			v := linear(x, stateDict[fmt.Sprintf("layer%d.attn_wv", li)])
			keys[li] = append(keys[li], k)
			values[li] = append(values[li], v)

			xAttn := make([]*Value, 0, nEmbd)
			for h := 0; h < nHead; h++ {
				hs := h * headDim
				qH := q[hs : hs+headDim]
				kH := make([][]*Value, len(keys[li]))
				vH := make([][]*Value, len(values[li]))
				for t := 0; t < len(keys[li]); t++ {
					kH[t] = keys[li][t][hs : hs+headDim]
					vH[t] = values[li][t][hs : hs+headDim]
				}

				// Compute attention logits: dot(q_h, k_h[t]) / sqrt(head_dim)
				scale := math.Sqrt(float64(headDim))
				attnLogits := make([]*Value, len(kH))
				for t := 0; t < len(kH); t++ {
					dot := qH[0].Mul(kH[t][0])
					for j := 1; j < headDim; j++ {
						dot = dot.Add(qH[j].Mul(kH[t][j]))
					}
					attnLogits[t] = dot.MulScalar(1.0 / scale)
				}

				attnWeights := softmax(attnLogits)

				// Weighted sum of value vectors
				headOut := make([]*Value, headDim)
				for j := 0; j < headDim; j++ {
					sum := attnWeights[0].Mul(vH[0][j])
					for t := 1; t < len(vH); t++ {
						sum = sum.Add(attnWeights[t].Mul(vH[t][j]))
					}
					headOut[j] = sum
				}
				xAttn = append(xAttn, headOut...)
			}

			x = linear(xAttn, stateDict[fmt.Sprintf("layer%d.attn_wo", li)])
			for i := 0; i < nEmbd; i++ {
				x[i] = x[i].Add(xResidual[i])
			}

			// 2) MLP block
			xResidual = x
			x = rmsnorm(x)
			x = linear(x, stateDict[fmt.Sprintf("layer%d.mlp_fc1", li)])
			for i := range x {
				x[i] = x[i].ReLU()
			}
			x = linear(x, stateDict[fmt.Sprintf("layer%d.mlp_fc2", li)])
			for i := 0; i < nEmbd; i++ {
				x[i] = x[i].Add(xResidual[i])
			}
		}

		logits := linear(x, stateDict["lm_head"])
		return logits
	}

	// --- Adam Optimizer ---
	// Let there be Adam, the blessed optimizer and its buffers
	learningRate := 0.01
	beta1 := 0.85
	beta2 := 0.99
	epsAdam := 1e-8
	mBuf := make([]float64, len(params)) // first moment buffer
	vBuf := make([]float64, len(params)) // second moment buffer

	// --- Training Loop ---
	numSteps := 1000
	for step := 0; step < numSteps; step++ {
		// Take single document, tokenize it, surround it with BOS on both sides
		doc := docs[step%len(docs)]
		tokens := make([]int, 0, len(doc)+2)
		tokens = append(tokens, BOS)
		for _, ch := range doc {
			tokens = append(tokens, charToIdx[ch])
		}
		tokens = append(tokens, BOS)

		n := blockSize
		if len(tokens)-1 < n {
			n = len(tokens) - 1
		}

		// Forward pass: build computation graph all the way to the loss
		keys := make([][][]*Value, nLayer)
		vals := make([][][]*Value, nLayer)
		for i := 0; i < nLayer; i++ {
			keys[i] = make([][]*Value, 0)
			vals[i] = make([][]*Value, 0)
		}

		losses := make([]*Value, 0, n)
		for posID := 0; posID < n; posID++ {
			tokenID := tokens[posID]
			targetID := tokens[posID+1]
			logits := gpt(tokenID, posID, keys, vals)
			probs := softmax(logits)
			lossT := probs[targetID].Log().Neg()
			losses = append(losses, lossT)
		}

		// Final average loss over the document sequence. May yours be low.
		loss := losses[0]
		for i := 1; i < len(losses); i++ {
			loss = loss.Add(losses[i])
		}
		loss = loss.MulScalar(1.0 / float64(n))

		// Backward: calculate gradients with respect to all model parameters
		loss.Backward()

		// Adam optimizer update
		lrT := learningRate * (1.0 - float64(step)/float64(numSteps))
		for i, p := range params {
			mBuf[i] = beta1*mBuf[i] + (1-beta1)*p.grad
			vBuf[i] = beta2*vBuf[i] + (1-beta2)*p.grad*p.grad
			mHat := mBuf[i] / (1 - math.Pow(beta1, float64(step+1)))
			vHat := vBuf[i] / (1 - math.Pow(beta2, float64(step+1)))
			p.data -= lrT * mHat / (math.Sqrt(vHat) + epsAdam)
			p.grad = 0
		}

		fmt.Printf("\rstep %4d / %4d | loss %.4f", step+1, numSteps, loss.data)
	}

	// --- Inference ---
	// May the model babble back to us
	temperature := 0.5 // in (0, 1], control the "creativity" of generated text
	fmt.Println("\n--- inference (new, hallucinated names) ---")
	for sampleIdx := 0; sampleIdx < 20; sampleIdx++ {
		keys := make([][][]*Value, nLayer)
		vals := make([][][]*Value, nLayer)
		for i := 0; i < nLayer; i++ {
			keys[i] = make([][]*Value, 0)
			vals[i] = make([][]*Value, 0)
		}

		tokenID := BOS
		var sample []rune
		for posID := 0; posID < blockSize; posID++ {
			logits := gpt(tokenID, posID, keys, vals)
			// Apply temperature scaling
			scaledLogits := make([]*Value, len(logits))
			for i, l := range logits {
				scaledLogits[i] = l.MulScalar(1.0 / temperature)
			}
			probs := softmax(scaledLogits)
			weights := make([]float64, len(probs))
			for i, p := range probs {
				weights[i] = p.data
			}
			tokenID = weightedChoice(rng, weights)
			if tokenID == BOS {
				break
			}
			sample = append(sample, uchars[tokenID])
		}
		fmt.Printf("sample %2d: %s\n", sampleIdx+1, string(sample))
	}
}
