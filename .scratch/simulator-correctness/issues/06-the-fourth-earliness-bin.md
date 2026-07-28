# 06 — The fourth earliness bin

**What to build:** `counts_earliness[3]` is never assigned, so `general[10]` is
identically zero at every instant under every window. Closes B10.

`policies/feature_extraction.py:208-214,230`:

```python
counts_earliness = [0, 0, 0, 0]
if tau < 400: counts_earliness[0] = ...(earliness < 400)
if tau < 500: counts_earliness[1] = ...(400 <= earliness < 500)
if tau < 600: counts_earliness[2] = ...(500 <= earliness < 600)
# counts_earliness[3] is never assigned
general[7:11] = [count / self._number_clients for count in counts_earliness]
```

Three verified consequences: `W[10]` never updates because the gradient is
`lr * err * X` and `X[10]` is always zero, so **the effective model is
18-dimensional, not 19**; no counting bin counts a Client whose window opens at
minute 600 or later, which under the real `Uniform[300, 720]` is **28.7% of
demand**; and for `tau >= 600` all four bins are zero, so five of the twelve
general features carry no information across roughly half the horizon — exactly
where end-of-day pressure and delay penalties start to bite.

**Confirmed in the committed capture: `final_w[10] == 0.0` exactly.**

**Blocked by:** 01

**Status:** open

- [ ] Assign the fourth bin so it covers what the other three leave out, and
      `general[10]` stops being identically zero.
- [ ] **Do not redesign the cuts.** Where the boundaries should sit — and
      whether four bins keyed to absolute minutes is the right shape at all — is
      feature design, which belongs to the modeling effort along with B4 and B6.
      This ticket makes the existing design *work*; it does not replace it.
- [ ] Invariant: the four earliness bins **partition** pending demand — the four
      fractions sum to pending Clients over `number_clients`, at every tau. That
      property is what makes "a Client falls in no bin" impossible to reintroduce.
- [ ] Update the docstrings: they document the other dead weight (the `X[:,13]`
      filler) but not this one. Note that `W[13]` remains dead by design and
      `final_w[13] == 0.0` — after this ticket exactly one weight is
      deliberately dead, not two accidentally.

## Predicted self-golden diff

**Frozen-W block: exactly zero, and this is provable in advance.** The frozen W
is the committed `final_w`, whose entry 10 is `0.0` exactly. Feature 10 enters Q
only as `X[10] * W[10]`, so any change to `X[10]` multiplies by zero. Decisions
cannot move. Every metric on every seed must be bit-identical.

If the frozen-W block moves at all, the change escaped feature 10 — most likely
the fourth bin's definition also altered bins 0–2, which this ticket forbids.
That is the check.

**Training block: divergence, from the first weight update onward.** `W[10]`
starts receiving gradient, so the trained W is 19-dimensional for the first time
and every subsequent decision can differ. **Final-eval block: divergence**, since
it runs on that changed final W.

Direction is not predicted. A feature that has never been trained carries no
prior about whether it helps; the honest expectation is "the W trajectory
changes shape", not "cost improves".

## Evidence required

Frozen-W block bit-identical (the strong claim). `final_w[10] != 0.0` after
training — the point of the ticket is that the weight now trains. Partition
invariant green. The 60-seed bench before/after, reported without a directional
claim.

## Comments
