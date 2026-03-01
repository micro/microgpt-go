package main

import (
	"math"
	"math/rand"
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
