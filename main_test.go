package main

import (
	"math"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestValueAdd(t *testing.T) {
	a := NewValue(2.0)
	b := NewValue(3.0)
	c := a.Add(b)
	if c.data != 5.0 {
		t.Errorf("expected 5.0, got %f", c.data)
	}
}

func TestValueMul(t *testing.T) {
	a := NewValue(2.0)
	b := NewValue(3.0)
	c := a.Mul(b)
	if c.data != 6.0 {
		t.Errorf("expected 6.0, got %f", c.data)
	}
}

func TestValuePow(t *testing.T) {
	a := NewValue(3.0)
	b := a.Pow(2.0)
	if math.Abs(b.data-9.0) > 1e-10 {
		t.Errorf("expected 9.0, got %f", b.data)
	}
	b.Backward()
	// d/da (a^2) = 2a = 6.0
	if math.Abs(a.grad-6.0) > 1e-10 {
		t.Errorf("expected grad 6.0, got %f", a.grad)
	}
}

func TestValueLog(t *testing.T) {
	a := NewValue(math.E)
	b := a.Log()
	if math.Abs(b.data-1.0) > 1e-10 {
		t.Errorf("expected 1.0, got %f", b.data)
	}
	b.Backward()
	// d/da (ln(a)) = 1/a = 1/e
	if math.Abs(a.grad-1.0/math.E) > 1e-10 {
		t.Errorf("expected grad %f, got %f", 1.0/math.E, a.grad)
	}
}

func TestValueExp(t *testing.T) {
	a := NewValue(1.0)
	b := a.Exp()
	if math.Abs(b.data-math.E) > 1e-10 {
		t.Errorf("expected %f, got %f", math.E, b.data)
	}
	b.Backward()
	// d/da (exp(a)) = exp(a) = e
	if math.Abs(a.grad-math.E) > 1e-10 {
		t.Errorf("expected grad %f, got %f", math.E, a.grad)
	}
}

func TestValueReLU(t *testing.T) {
	a := NewValue(3.0)
	b := a.ReLU()
	if b.data != 3.0 {
		t.Errorf("expected 3.0, got %f", b.data)
	}
	b.Backward()
	if a.grad != 1.0 {
		t.Errorf("expected grad 1.0 for positive input, got %f", a.grad)
	}

	c := NewValue(-2.0)
	d := c.ReLU()
	if d.data != 0.0 {
		t.Errorf("expected 0.0, got %f", d.data)
	}
	d.Backward()
	if c.grad != 0.0 {
		t.Errorf("expected grad 0.0 for negative input, got %f", c.grad)
	}
}

func TestValueNeg(t *testing.T) {
	a := NewValue(5.0)
	b := a.Neg()
	if b.data != -5.0 {
		t.Errorf("expected -5.0, got %f", b.data)
	}
}

func TestValueSub(t *testing.T) {
	a := NewValue(5.0)
	b := NewValue(3.0)
	c := a.Sub(b)
	if math.Abs(c.data-2.0) > 1e-10 {
		t.Errorf("expected 2.0, got %f", c.data)
	}
}

func TestValueDiv(t *testing.T) {
	a := NewValue(6.0)
	b := NewValue(3.0)
	c := a.Div(b)
	if math.Abs(c.data-2.0) > 1e-10 {
		t.Errorf("expected 2.0, got %f", c.data)
	}
}

func TestBackwardSimple(t *testing.T) {
	// f(a, b) = a * b + a
	// df/da = b + 1 = 4.0
	// df/db = a = 2.0
	a := NewValue(2.0)
	b := NewValue(3.0)
	c := a.Mul(b)
	d := c.Add(a)
	d.Backward()
	if math.Abs(a.grad-4.0) > 1e-10 {
		t.Errorf("expected grad 4.0 for a, got %f", a.grad)
	}
	if math.Abs(b.grad-2.0) > 1e-10 {
		t.Errorf("expected grad 2.0 for b, got %f", b.grad)
	}
}

func TestBackwardComplex(t *testing.T) {
	// f(x) = exp(x^2 + x)
	// f'(x) = (2x + 1) * exp(x^2 + x)
	// At x = 1: f'(1) = 3 * exp(2)
	x := NewValue(1.0)
	x2 := x.Mul(x)
	sum := x2.Add(x)
	result := sum.Exp()
	result.Backward()
	expected := 3.0 * math.Exp(2.0)
	if math.Abs(x.grad-expected) > 1e-6 {
		t.Errorf("expected grad %f, got %f", expected, x.grad)
	}
}

func TestSoftmax(t *testing.T) {
	vals := []*Value{NewValue(1.0), NewValue(2.0), NewValue(3.0)}
	probs := softmax(vals)

	// Probabilities should sum to 1
	sum := 0.0
	for _, p := range probs {
		sum += p.data
	}
	if math.Abs(sum-1.0) > 1e-6 {
		t.Errorf("softmax probabilities should sum to 1, got %f", sum)
	}

	// Each probability should be positive
	for i, p := range probs {
		if p.data <= 0 {
			t.Errorf("softmax probability %d should be positive, got %f", i, p.data)
		}
	}

	// Higher logit should produce higher probability
	if probs[2].data <= probs[1].data || probs[1].data <= probs[0].data {
		t.Errorf("softmax probabilities should be monotonically increasing with logits")
	}
}

func TestSoftmaxBackward(t *testing.T) {
	// Test that gradients flow through softmax
	vals := []*Value{NewValue(1.0), NewValue(2.0), NewValue(3.0)}
	probs := softmax(vals)
	// Take the log of the first probability (like cross-entropy loss)
	loss := probs[0].Log().Neg()
	loss.Backward()
	// All input values should have non-zero gradients
	for i, v := range vals {
		if v.grad == 0 {
			t.Errorf("expected non-zero gradient for input %d", i)
		}
	}
}

func TestRMSNorm(t *testing.T) {
	vals := []*Value{NewValue(1.0), NewValue(2.0), NewValue(3.0)}
	normed := rmsnorm(vals)

	if len(normed) != 3 {
		t.Fatalf("expected 3 outputs, got %d", len(normed))
	}

	// Check that the output is finite
	for i, v := range normed {
		if math.IsNaN(v.data) || math.IsInf(v.data, 0) {
			t.Errorf("rmsnorm output %d is NaN or Inf", i)
		}
	}

	// Verify RMS normalization: output = x / sqrt(mean(x^2) + eps)
	// mean(x^2) = (1 + 4 + 9) / 3 = 14/3
	// scale = 1 / sqrt(14/3 + 1e-5)
	ms := (1.0 + 4.0 + 9.0) / 3.0
	scale := 1.0 / math.Sqrt(ms+1e-5)
	for i, v := range normed {
		expected := float64(i+1) * scale
		if math.Abs(v.data-expected) > 1e-6 {
			t.Errorf("rmsnorm output %d: expected %f, got %f", i, expected, v.data)
		}
	}
}

func TestLinear(t *testing.T) {
	// 2x3 matrix * 3-vector = 2-vector
	w := [][]*Value{
		{NewValue(1), NewValue(2), NewValue(3)},
		{NewValue(4), NewValue(5), NewValue(6)},
	}
	x := []*Value{NewValue(1), NewValue(1), NewValue(1)}
	y := linear(x, w)
	if len(y) != 2 {
		t.Fatalf("expected 2 outputs, got %d", len(y))
	}
	// [1,2,3] . [1,1,1] = 6
	if math.Abs(y[0].data-6.0) > 1e-10 {
		t.Errorf("expected 6.0, got %f", y[0].data)
	}
	// [4,5,6] . [1,1,1] = 15
	if math.Abs(y[1].data-15.0) > 1e-10 {
		t.Errorf("expected 15.0, got %f", y[1].data)
	}
}

func TestLinearBackward(t *testing.T) {
	// y = w * x, compute gradient of sum(y) w.r.t. x
	w := [][]*Value{
		{NewValue(2), NewValue(3)},
	}
	x := []*Value{NewValue(1), NewValue(1)}
	y := linear(x, w)
	y[0].Backward()
	// dy/dx[0] = w[0][0] = 2, dy/dx[1] = w[0][1] = 3
	if math.Abs(x[0].grad-2.0) > 1e-10 {
		t.Errorf("expected grad 2.0 for x[0], got %f", x[0].grad)
	}
	if math.Abs(x[1].grad-3.0) > 1e-10 {
		t.Errorf("expected grad 3.0 for x[1], got %f", x[1].grad)
	}
}

func TestWeightedChoice(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	weights := []float64{0.1, 0.2, 0.7}
	counts := make([]int, 3)
	n := 10000
	for i := 0; i < n; i++ {
		idx := weightedChoice(rng, weights)
		counts[idx]++
	}
	// Check that the distribution roughly matches the weights
	for i, w := range weights {
		expected := w * float64(n)
		actual := float64(counts[i])
		if math.Abs(actual-expected)/expected > 0.1 { // 10% tolerance
			t.Errorf("weightedChoice index %d: expected ~%.0f, got %.0f", i, expected, actual)
		}
	}
}

func TestMatrix(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	m := matrix(rng, 3, 4, 0.08)
	if len(m) != 3 {
		t.Fatalf("expected 3 rows, got %d", len(m))
	}
	for i, row := range m {
		if len(row) != 4 {
			t.Fatalf("expected 4 columns in row %d, got %d", i, len(row))
		}
	}
	// Check that values are approximately in the right range for std=0.08
	for _, row := range m {
		for _, v := range row {
			if math.Abs(v.data) > 1.0 { // very unlikely for std=0.08
				t.Errorf("matrix value %f seems too large for std=0.08", v.data)
			}
		}
	}
}

func TestBuildTokenizer(t *testing.T) {
	docs := []string{"hello", "world"}
	uchars, charToIdx := buildTokenizer(docs)

	// Should contain all unique characters
	expected := map[rune]bool{'h': true, 'e': true, 'l': true, 'o': true, 'w': true, 'r': true, 'd': true}
	if len(uchars) != len(expected) {
		t.Errorf("expected %d unique chars, got %d", len(expected), len(uchars))
	}
	for _, ch := range uchars {
		if !expected[ch] {
			t.Errorf("unexpected char %c in uchars", ch)
		}
	}

	// Should be sorted
	for i := 1; i < len(uchars); i++ {
		if uchars[i] <= uchars[i-1] {
			t.Errorf("uchars not sorted: %c <= %c", uchars[i], uchars[i-1])
		}
	}

	// charToIdx should map correctly
	for i, ch := range uchars {
		if charToIdx[ch] != i {
			t.Errorf("charToIdx[%c] = %d, expected %d", ch, charToIdx[ch], i)
		}
	}
}

func TestReadLines(t *testing.T) {
	input := "hello\n  world  \n\n  foo  \n"
	r := strings.NewReader(input)
	docs, err := readLines(r)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(docs) != 3 {
		t.Fatalf("expected 3 docs, got %d", len(docs))
	}
	if docs[0] != "hello" || docs[1] != "world" || docs[2] != "foo" {
		t.Errorf("unexpected docs: %v", docs)
	}
}

func TestParamKeys(t *testing.T) {
	keys := paramKeys(2)
	expected := []string{
		"wte", "wpe", "lm_head",
		"layer0.attn_wq", "layer0.attn_wk", "layer0.attn_wv", "layer0.attn_wo", "layer0.mlp_fc1", "layer0.mlp_fc2",
		"layer1.attn_wq", "layer1.attn_wk", "layer1.attn_wv", "layer1.attn_wo", "layer1.mlp_fc1", "layer1.mlp_fc2",
	}
	if len(keys) != len(expected) {
		t.Fatalf("expected %d keys, got %d", len(expected), len(keys))
	}
	for i, k := range keys {
		if k != expected[i] {
			t.Errorf("key %d: expected %s, got %s", i, expected[i], k)
		}
	}
}

func TestInitWeights(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	stateDict := initWeights(rng, 5, 1, 4, 4)

	// Check that all expected keys exist
	for _, key := range paramKeys(1) {
		if _, ok := stateDict[key]; !ok {
			t.Errorf("missing key %s in stateDict", key)
		}
	}

	// Check wte dimensions
	if len(stateDict["wte"]) != 5 || len(stateDict["wte"][0]) != 4 {
		t.Errorf("wte has wrong dimensions: %dx%d", len(stateDict["wte"]), len(stateDict["wte"][0]))
	}
}

func TestFlattenParams(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	stateDict := initWeights(rng, 5, 1, 4, 4)
	params := flattenParams(stateDict, 1)

	// Expected params: wte(5*4) + wpe(4*4) + lm_head(5*4) + attn_wq(4*4) + attn_wk(4*4) + attn_wv(4*4) + attn_wo(4*4) + mlp_fc1(16*4) + mlp_fc2(4*16)
	expected := 5*4 + 4*4 + 5*4 + 4*4 + 4*4 + 4*4 + 4*4 + 4*16 + 4*16
	if len(params) != expected {
		t.Errorf("expected %d params, got %d", expected, len(params))
	}
}

func TestSaveLoadModel(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	nLayer, nEmbd, blockSize, nHead := 1, 4, 4, 2
	vocabSize := 5
	stateDict := initWeights(rng, vocabSize, nLayer, nEmbd, blockSize)
	uchars := []rune{'a', 'b', 'c', 'd'}

	path := t.TempDir() + "/model.json"
	err := saveModel(path, stateDict, uchars, nLayer, nEmbd, blockSize, nHead)
	if err != nil {
		t.Fatalf("saveModel error: %v", err)
	}

	loaded, loadedUchars, loadedCharToIdx, lNLayer, lNEmbd, lBlockSize, lNHead, err := loadModel(path)
	if err != nil {
		t.Fatalf("loadModel error: %v", err)
	}

	if lNLayer != nLayer || lNEmbd != nEmbd || lBlockSize != blockSize || lNHead != nHead {
		t.Errorf("hyperparams mismatch: got (%d,%d,%d,%d), expected (%d,%d,%d,%d)",
			lNLayer, lNEmbd, lBlockSize, lNHead, nLayer, nEmbd, blockSize, nHead)
	}

	if len(loadedUchars) != len(uchars) {
		t.Fatalf("vocab size mismatch: got %d, expected %d", len(loadedUchars), len(uchars))
	}
	for i, ch := range loadedUchars {
		if ch != uchars[i] {
			t.Errorf("uchars[%d]: got %c, expected %c", i, ch, uchars[i])
		}
	}

	// Verify charToIdx is correct
	for i, ch := range loadedUchars {
		if loadedCharToIdx[ch] != i {
			t.Errorf("charToIdx[%c] = %d, expected %d", ch, loadedCharToIdx[ch], i)
		}
	}

	// Verify weights match
	for _, key := range paramKeys(nLayer) {
		orig := stateDict[key]
		load := loaded[key]
		if len(orig) != len(load) {
			t.Fatalf("key %s: row count mismatch %d vs %d", key, len(orig), len(load))
		}
		for i := range orig {
			for j := range orig[i] {
				if math.Abs(orig[i][j].data-load[i][j].data) > 1e-10 {
					t.Errorf("key %s[%d][%d]: %.10f vs %.10f", key, i, j, orig[i][j].data, load[i][j].data)
				}
			}
		}
	}
}

func TestMakeGPT(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	nLayer, nEmbd, blockSize, nHead := 1, 4, 4, 2
	vocabSize := 5
	stateDict := initWeights(rng, vocabSize, nLayer, nEmbd, blockSize)

	gpt := makeGPT(stateDict, nLayer, nEmbd, nHead)

	keys := make([][][]*Value, nLayer)
	vals := make([][][]*Value, nLayer)
	for i := 0; i < nLayer; i++ {
		keys[i] = make([][]*Value, 0)
		vals[i] = make([][]*Value, 0)
	}

	logits := gpt(0, 0, keys, vals)
	if len(logits) != vocabSize {
		t.Errorf("expected %d logits, got %d", vocabSize, len(logits))
	}
	for i, l := range logits {
		if math.IsNaN(l.data) || math.IsInf(l.data, 0) {
			t.Errorf("logit %d is NaN or Inf", i)
		}
	}
}

func TestGenerateSample(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	nLayer, nEmbd, blockSize, nHead := 1, 4, 4, 2
	vocabSize := 5
	stateDict := initWeights(rng, vocabSize, nLayer, nEmbd, blockSize)
	uchars := []rune{'a', 'b', 'c', 'd'}
	BOS := len(uchars)

	gpt := makeGPT(stateDict, nLayer, nEmbd, nHead)
	sample := generateSample(rng, gpt, uchars, BOS, nLayer, blockSize, 0.5)

	if len(sample) > blockSize {
		t.Errorf("sample length %d exceeds block size %d", len(sample), blockSize)
	}
	// All characters in output should be from the vocabulary
	validChars := map[rune]bool{'a': true, 'b': true, 'c': true, 'd': true}
	for _, ch := range sample {
		if !validChars[ch] {
			t.Errorf("sample contains invalid character %c", ch)
		}
	}
}

func TestGenerateSampleWithPrompt(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	nLayer, nEmbd, blockSize, nHead := 1, 4, 4, 2
	vocabSize := 5
	stateDict := initWeights(rng, vocabSize, nLayer, nEmbd, blockSize)
	uchars := []rune{'a', 'b', 'c', 'd'}
	charToIdx := map[rune]int{'a': 0, 'b': 1, 'c': 2, 'd': 3}
	BOS := len(uchars)

	gpt := makeGPT(stateDict, nLayer, nEmbd, nHead)
	sample := generateSampleWithPrompt(rng, gpt, uchars, charToIdx, BOS, nLayer, blockSize, 0.5, "ab", blockSize)

	// Should start with the prompt
	if len(sample) < 2 || sample[:2] != "ab" {
		t.Errorf("expected sample to start with 'ab', got '%s'", sample)
	}
}

func TestTrainModel(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	docs := []string{"ab", "cd", "ab", "cd"}
	uchars, charToIdx := buildTokenizer(docs)
	BOS := len(uchars)
	vocabSize := len(uchars) + 1

	nLayer, nEmbd, blockSize, nHead := 1, 4, 4, 2
	stateDict := initWeights(rng, vocabSize, nLayer, nEmbd, blockSize)
	params := flattenParams(stateDict, nLayer)
	gpt := makeGPT(stateDict, nLayer, nEmbd, nHead)

	// Run a few training steps
	loss := trainModel(gpt, params, docs, charToIdx, BOS, nLayer, blockSize, 5)

	// Loss should be a finite positive number
	if math.IsNaN(loss) || math.IsInf(loss, 0) || loss < 0 {
		t.Errorf("unexpected loss value: %f", loss)
	}
}

func TestWellKnownDatasetsRegistry(t *testing.T) {
	// Every entry must have Name, URL, Category, and Description.
	if len(wellKnownDatasets) == 0 {
		t.Fatal("wellKnownDatasets registry is empty")
	}
	for key, ds := range wellKnownDatasets {
		if ds.Name != key {
			t.Errorf("dataset %q: Name %q does not match map key", key, ds.Name)
		}
		if ds.URL == "" {
			t.Errorf("dataset %q: URL is empty", key)
		}
		if ds.Category == "" {
			t.Errorf("dataset %q: Category is empty", key)
		}
		if ds.Description == "" {
			t.Errorf("dataset %q: Description is empty", key)
		}
	}
}

func TestDatasetCacheDir(t *testing.T) {
	dir, err := datasetCacheDir()
	if err != nil {
		t.Fatalf("datasetCacheDir: %v", err)
	}
	if dir == "" {
		t.Fatal("datasetCacheDir returned empty string")
	}
	info, err := os.Stat(dir)
	if err != nil {
		t.Fatalf("cache dir does not exist: %v", err)
	}
	if !info.IsDir() {
		t.Fatalf("cache dir is not a directory: %s", dir)
	}
}

func TestDownloadToFile(t *testing.T) {
	// Set up a local HTTP server to avoid real network calls in tests.
	content := "alice\nbob\ncharlie\n"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(content))
	}))
	defer srv.Close()

	dst := filepath.Join(t.TempDir(), "test_download.txt")
	if err := downloadToFile(srv.URL, dst); err != nil {
		t.Fatalf("downloadToFile: %v", err)
	}

	got, err := os.ReadFile(dst)
	if err != nil {
		t.Fatalf("reading downloaded file: %v", err)
	}
	if string(got) != content {
		t.Errorf("downloaded content mismatch: got %q, want %q", string(got), content)
	}
}

func TestDownloadToFileHTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "not found", http.StatusNotFound)
	}))
	defer srv.Close()

	dst := filepath.Join(t.TempDir(), "should_not_exist.txt")
	err := downloadToFile(srv.URL, dst)
	if err == nil {
		t.Fatal("expected error for HTTP 404, got nil")
	}
	if !strings.Contains(err.Error(), "404") {
		t.Errorf("error should mention HTTP status: %v", err)
	}
	// Partial file should have been cleaned up
	if _, err := os.Stat(dst); !os.IsNotExist(err) {
		t.Error("partial file should have been removed after HTTP error")
	}
}

func TestResolveDatasetWellKnown(t *testing.T) {
	// Set up a local test server with fake data.
	content := "alpha\nbeta\ngamma\n"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(content))
	}))
	defer srv.Close()

	// Temporarily add a test dataset to the registry.
	wellKnownDatasets["_test_resolve"] = WellKnownDataset{
		Name:        "_test_resolve",
		Category:    "test",
		Description: "test dataset",
		URL:         srv.URL + "/test.txt",
	}
	defer delete(wellKnownDatasets, "_test_resolve")

	// Clean up any cached file first.
	cacheDir, err := datasetCacheDir()
	if err != nil {
		t.Fatalf("datasetCacheDir: %v", err)
	}
	cached := filepath.Join(cacheDir, "_test_resolve.txt")
	os.Remove(cached) // ignore error; file may not exist
	t.Cleanup(func() { os.Remove(cached) })

	// First call should download.
	path, err := resolveDataset("_test_resolve")
	if err != nil {
		t.Fatalf("resolveDataset (first): %v", err)
	}
	if path != cached {
		t.Errorf("expected cached path %s, got %s", cached, path)
	}

	// File should exist with correct content.
	got, err := os.ReadFile(cached)
	if err != nil {
		t.Fatalf("reading cached file: %v", err)
	}
	if string(got) != content {
		t.Errorf("cached content mismatch: got %q, want %q", string(got), content)
	}

	// Second call should hit cache (no download needed).
	path2, err := resolveDataset("_test_resolve")
	if err != nil {
		t.Fatalf("resolveDataset (cached): %v", err)
	}
	if path2 != cached {
		t.Errorf("expected same cached path on second call")
	}
}

func TestResolveDatasetLiteralPath(t *testing.T) {
	// A path that doesn't match any well-known name should be returned as-is.
	path, err := resolveDataset("/some/file/path.txt")
	if err != nil {
		t.Fatalf("resolveDataset: %v", err)
	}
	if path != "/some/file/path.txt" {
		t.Errorf("expected literal path, got %s", path)
	}
}

func TestListDatasets(t *testing.T) {
	// Just ensure listDatasets doesn't panic; output goes to stdout.
	listDatasets()
}
