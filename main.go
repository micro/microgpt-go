// The most atomic way to train and run inference for a GPT in pure, dependency-free Go.
// This file is the complete algorithm.
// Everything else is just efficiency.
//
// Translated from the Python original by @karpathy:
// https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
package main

import (
	"bufio"
	"encoding/json"
	"flag"
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

// --- Model Persistence ---

// ModelData holds everything needed to save and restore a trained model.
type ModelData struct {
	Vocab     []string            `json:"vocab"`
	NLayer    int                 `json:"n_layer"`
	NEmbd     int                 `json:"n_embd"`
	BlockSize int                 `json:"block_size"`
	NHead     int                 `json:"n_head"`
	Weights   map[string][][]float64 `json:"weights"`
}

// paramKeys returns the deterministic key order for model parameters.
func paramKeys(nLayer int) []string {
	keys := []string{"wte", "wpe", "lm_head"}
	for i := 0; i < nLayer; i++ {
		keys = append(keys,
			fmt.Sprintf("layer%d.attn_wq", i),
			fmt.Sprintf("layer%d.attn_wk", i),
			fmt.Sprintf("layer%d.attn_wv", i),
			fmt.Sprintf("layer%d.attn_wo", i),
			fmt.Sprintf("layer%d.mlp_fc1", i),
			fmt.Sprintf("layer%d.mlp_fc2", i),
		)
	}
	return keys
}

// flattenParams collects all model parameters in deterministic order.
func flattenParams(stateDict map[string][][]*Value, nLayer int) []*Value {
	var params []*Value
	for _, key := range paramKeys(nLayer) {
		for _, row := range stateDict[key] {
			params = append(params, row...)
		}
	}
	return params
}

// saveModel writes model weights and vocabulary to a JSON file.
func saveModel(path string, stateDict map[string][][]*Value, uchars []rune, nLayer, nEmbd, blockSize, nHead int) error {
	weights := make(map[string][][]float64)
	for _, key := range paramKeys(nLayer) {
		mat := stateDict[key]
		fmat := make([][]float64, len(mat))
		for i, row := range mat {
			frow := make([]float64, len(row))
			for j, v := range row {
				frow[j] = v.data
			}
			fmat[i] = frow
		}
		weights[key] = fmat
	}

	vocab := make([]string, len(uchars))
	for i, ch := range uchars {
		vocab[i] = string(ch)
	}

	md := ModelData{
		Vocab:     vocab,
		NLayer:    nLayer,
		NEmbd:     nEmbd,
		BlockSize: blockSize,
		NHead:     nHead,
		Weights:   weights,
	}

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	return enc.Encode(md)
}

// loadModel reads model weights and vocabulary from a JSON file.
func loadModel(path string) (map[string][][]*Value, []rune, map[rune]int, int, int, int, int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, nil, 0, 0, 0, 0, err
	}
	defer f.Close()

	var md ModelData
	if err := json.NewDecoder(f).Decode(&md); err != nil {
		return nil, nil, nil, 0, 0, 0, 0, err
	}

	uchars := make([]rune, len(md.Vocab))
	charToIdx := make(map[rune]int)
	for i, s := range md.Vocab {
		r := []rune(s)
		uchars[i] = r[0]
		charToIdx[r[0]] = i
	}

	stateDict := make(map[string][][]*Value)
	for key, fmat := range md.Weights {
		mat := make([][]*Value, len(fmat))
		for i, frow := range fmat {
			row := make([]*Value, len(frow))
			for j, v := range frow {
				row[j] = NewValue(v)
			}
			mat[i] = row
		}
		stateDict[key] = mat
	}

	return stateDict, uchars, charToIdx, md.NLayer, md.NEmbd, md.BlockSize, md.NHead, nil
}

// readLines reads non-empty trimmed lines from an io.Reader.
func readLines(r io.Reader) ([]string, error) {
	var docs []string
	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line != "" {
			docs = append(docs, line)
		}
	}
	return docs, scanner.Err()
}

// downloadDefaultDataset downloads the names dataset if input.txt doesn't exist.
func downloadDefaultDataset(path string) error {
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		return nil
	}
	namesURL := "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"
	resp, err := http.Get(namesURL) // #nosec G107 -- URL is a hardcoded constant
	if err != nil {
		return fmt.Errorf("downloading dataset: %w", err)
	}
	defer resp.Body.Close()
	out, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("creating %s: %w", path, err)
	}
	if _, err = io.Copy(out, resp.Body); err != nil {
		out.Close()
		return fmt.Errorf("writing %s: %w", path, err)
	}
	return out.Close()
}

// buildTokenizer creates a character-level tokenizer from the given documents.
func buildTokenizer(docs []string) ([]rune, map[rune]int) {
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
	return uchars, charToIdx
}

// initWeights creates randomly initialized model parameters.
func initWeights(rng *rand.Rand, vocabSize, nLayer, nEmbd, blockSize int) map[string][][]*Value {
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
	return stateDict
}

// makeGPT returns the GPT forward function for the given model configuration.
func makeGPT(stateDict map[string][][]*Value, nLayer, nEmbd, nHead int) func(int, int, [][][]*Value, [][][]*Value) []*Value {
	headDim := nEmbd / nHead
	return func(tokenID, posID int, keys, values [][][]*Value) []*Value {
		tokEmb := stateDict["wte"][tokenID]
		posEmb := stateDict["wpe"][posID]
		x := make([]*Value, nEmbd)
		for i := 0; i < nEmbd; i++ {
			x[i] = tokEmb[i].Add(posEmb[i])
		}
		x = rmsnorm(x)

		for li := 0; li < nLayer; li++ {
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
}

// trainModel runs the training loop and returns the final loss.
func trainModel(gpt func(int, int, [][][]*Value, [][][]*Value) []*Value, params []*Value, docs []string, charToIdx map[rune]int, BOS, nLayer, blockSize, numSteps int) float64 {
	learningRate := 0.01
	beta1 := 0.85
	beta2 := 0.99
	epsAdam := 1e-8
	mBuf := make([]float64, len(params))
	vBuf := make([]float64, len(params))

	var finalLoss float64
	for step := 0; step < numSteps; step++ {
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

		loss := losses[0]
		for i := 1; i < len(losses); i++ {
			loss = loss.Add(losses[i])
		}
		loss = loss.MulScalar(1.0 / float64(n))

		loss.Backward()

		lrT := learningRate * (1.0 - float64(step)/float64(numSteps))
		for i, p := range params {
			mBuf[i] = beta1*mBuf[i] + (1-beta1)*p.grad
			vBuf[i] = beta2*vBuf[i] + (1-beta2)*p.grad*p.grad
			mHat := mBuf[i] / (1 - math.Pow(beta1, float64(step+1)))
			vHat := vBuf[i] / (1 - math.Pow(beta2, float64(step+1)))
			p.data -= lrT * mHat / (math.Sqrt(vHat) + epsAdam)
			p.grad = 0
		}

		finalLoss = loss.data
		fmt.Printf("\rstep %4d / %4d | loss %.4f", step+1, numSteps, loss.data)
	}
	fmt.Println()
	return finalLoss
}

// generateSample produces a single sample from the model.
func generateSample(rng *rand.Rand, gpt func(int, int, [][][]*Value, [][][]*Value) []*Value, uchars []rune, BOS, nLayer, blockSize int, temperature float64) string {
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
	return string(sample)
}

// generateSampleWithPrompt produces a sample seeded with a prompt string.
func generateSampleWithPrompt(rng *rand.Rand, gpt func(int, int, [][][]*Value, [][][]*Value) []*Value, uchars []rune, charToIdx map[rune]int, BOS, nLayer, blockSize int, temperature float64, prompt string, maxTokens int) string {
	keys := make([][][]*Value, nLayer)
	vals := make([][][]*Value, nLayer)
	for i := 0; i < nLayer; i++ {
		keys[i] = make([][]*Value, 0)
		vals[i] = make([][]*Value, 0)
	}

	// Feed the prompt through the model
	tokenID := BOS
	posID := 0
	var result []rune
	for _, ch := range prompt {
		if idx, ok := charToIdx[ch]; ok && posID < blockSize-1 {
			_ = gpt(tokenID, posID, keys, vals)
			tokenID = idx
			result = append(result, ch)
			posID++
		}
	}

	// Generate continuation
	limit := maxTokens
	if limit <= 0 || limit > blockSize {
		limit = blockSize
	}
	for posID < limit {
		logits := gpt(tokenID, posID, keys, vals)
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
		result = append(result, uchars[tokenID])
		posID++
	}
	return string(result)
}

// --- HTTP Inference Server ---

type generateRequest struct {
	Prompt      string  `json:"prompt"`
	Temperature float64 `json:"temperature"`
	MaxTokens   int     `json:"max_tokens"`
}

type generateResponse struct {
	Text string `json:"text"`
}

func serveHTTP(addr string, rng *rand.Rand, gpt func(int, int, [][][]*Value, [][][]*Value) []*Value, uchars []rune, charToIdx map[rune]int, BOS, nLayer, blockSize int, defaultTemp float64) error {
	mux := http.NewServeMux()
	mux.HandleFunc("/generate", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "POST required", http.StatusMethodNotAllowed)
			return
		}
		var req generateRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "invalid JSON: "+err.Error(), http.StatusBadRequest)
			return
		}
		temp := req.Temperature
		if temp <= 0 || temp > 1 {
			temp = defaultTemp
		}
		maxTok := req.MaxTokens
		if maxTok <= 0 {
			maxTok = blockSize
		}

		text := generateSampleWithPrompt(rng, gpt, uchars, charToIdx, BOS, nLayer, blockSize, temp, req.Prompt, maxTok)

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(generateResponse{Text: text})
	})

	fmt.Printf("microgpt-go inference server listening on %s\n", addr)
	fmt.Printf("  POST /generate {\"prompt\": \"...\", \"temperature\": 0.5, \"max_tokens\": 16}\n")
	return http.ListenAndServe(addr, mux)
}

func main() {
	// --- CLI Flags ---
	dataset := flag.String("dataset", "input.txt", "path to training data (one entry per line), or \"-\" for stdin")
	numSteps := flag.Int("steps", 1000, "number of training steps")
	temperature := flag.Float64("temp", 0.5, "sampling temperature (0, 1]")
	numSamples := flag.Int("samples", 20, "number of samples to generate")
	savePath := flag.String("save", "", "save trained model weights to this JSON file")
	loadPath := flag.String("load", "", "load model weights from this JSON file (skip training)")
	mode := flag.String("mode", "train", "mode: train (train+infer), infer (generate only), serve (HTTP server)")
	addr := flag.String("addr", ":8080", "address for HTTP server (used with -mode serve)")
	flag.Parse()

	rng := rand.New(rand.NewSource(42)) // Let there be order among chaos

	var (
		stateDict map[string][][]*Value
		uchars    []rune
		charToIdx map[rune]int
		nLayer    = 1
		nEmbd     = 16
		blockSize = 16
		nHead     = 4
	)

	if *loadPath != "" {
		// Load model from file
		var err error
		stateDict, uchars, charToIdx, nLayer, nEmbd, blockSize, nHead, err = loadModel(*loadPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error loading model: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("loaded model from %s (vocab=%d, layers=%d, embd=%d)\n", *loadPath, len(uchars), nLayer, nEmbd)
	}

	BOS := len(uchars) // token id for Beginning of Sequence
	vocabSize := len(uchars) + 1

	if *mode == "train" || (*mode != "infer" && *mode != "serve") {
		// --- Dataset ---
		var docs []string
		if *dataset == "-" {
			// Read from stdin (streaming training)
			var err error
			docs, err = readLines(os.Stdin)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error reading stdin: %v\n", err)
				os.Exit(1)
			}
		} else {
			if err := downloadDefaultDataset(*dataset); err != nil {
				fmt.Fprintf(os.Stderr, "Error: %v\n", err)
				os.Exit(1)
			}
			file, err := os.Open(*dataset)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error opening %s: %v\n", *dataset, err)
				os.Exit(1)
			}
			docs, err = readLines(file)
			file.Close()
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error reading %s: %v\n", *dataset, err)
				os.Exit(1)
			}
		}
		rng.Shuffle(len(docs), func(i, j int) { docs[i], docs[j] = docs[j], docs[i] })
		fmt.Printf("num docs: %d\n", len(docs))

		if *loadPath == "" {
			// Build tokenizer from training data
			uchars, charToIdx = buildTokenizer(docs)
			BOS = len(uchars)
			vocabSize = len(uchars) + 1
			fmt.Printf("vocab size: %d\n", vocabSize)

			// Initialize fresh weights
			stateDict = initWeights(rng, vocabSize, nLayer, nEmbd, blockSize)
		}

		params := flattenParams(stateDict, nLayer)
		fmt.Printf("num params: %d\n", len(params))

		gpt := makeGPT(stateDict, nLayer, nEmbd, nHead)
		trainModel(gpt, params, docs, charToIdx, BOS, nLayer, blockSize, *numSteps)

		if *savePath != "" {
			if err := saveModel(*savePath, stateDict, uchars, nLayer, nEmbd, blockSize, nHead); err != nil {
				fmt.Fprintf(os.Stderr, "Error saving model: %v\n", err)
				os.Exit(1)
			}
			fmt.Printf("model saved to %s\n", *savePath)
		}

		// Inference after training
		fmt.Println("--- inference (new, hallucinated samples) ---")
		for sampleIdx := 0; sampleIdx < *numSamples; sampleIdx++ {
			text := generateSample(rng, gpt, uchars, BOS, nLayer, blockSize, *temperature)
			fmt.Printf("sample %2d: %s\n", sampleIdx+1, text)
		}
		return
	}

	// Infer or serve modes require a loaded model
	if *loadPath == "" {
		fmt.Fprintf(os.Stderr, "Error: -load is required for -mode %s\n", *mode)
		os.Exit(1)
	}

	gpt := makeGPT(stateDict, nLayer, nEmbd, nHead)

	if *mode == "serve" {
		if err := serveHTTP(*addr, rng, gpt, uchars, charToIdx, BOS, nLayer, blockSize, *temperature); err != nil {
			fmt.Fprintf(os.Stderr, "Server error: %v\n", err)
			os.Exit(1)
		}
		return
	}

	// mode == "infer"
	fmt.Println("--- inference (new, hallucinated samples) ---")
	for sampleIdx := 0; sampleIdx < *numSamples; sampleIdx++ {
		text := generateSample(rng, gpt, uchars, BOS, nLayer, blockSize, *temperature)
		fmt.Printf("sample %2d: %s\n", sampleIdx+1, text)
	}
}
