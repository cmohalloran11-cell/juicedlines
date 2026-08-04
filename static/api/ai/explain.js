// Vercel serverless function — grounded AI Juice explanation for ONE projection.
// The static site has no server, so this function holds the Gemini key and makes the call.
// It mirrors ai_juice.py: it may ONLY use the numbers passed in the request body — it cannot
// fabricate a stat. Set GEMINI_API_KEY (and optionally AI_MODEL) in Vercel → Project →
// Settings → Environment Variables. Runs on Vercel's Node runtime (global fetch, Node 18+).

const SYSTEM = `You are AI Juice, the analyst voice of the Juiced sports-prop research platform. You explain, in plain language, WHY a statistical projection landed where it did and HOW UNCERTAIN it is.

Hard rules — follow exactly:
1. Use ONLY the numbers in the DATA block. Never invent a statistic, rate, rank, matchup detail, or source that is not present there. If a fact isn't in DATA, say you don't have it.
2. Never state or imply an outcome is guaranteed. Frame everything in terms of probability and uncertainty. A projection is a distribution, not a prediction.
3. Explain what drives the number using the fields provided (sample size, floor/ceiling band, edge vs the line, confidence), and how wide the uncertainty is.
4. Be concise and concrete: 2-4 short sentences. No hype, no betting advice, no "lock"/"guaranteed"/"free money" language. This is research, not a pick.
5. If the data is thin (small sample), say the projection is low-confidence.`;

module.exports = async function handler(req, res) {
  const key = process.env.GEMINI_API_KEY;
  const model = process.env.AI_MODEL || "gemini-2.5-flash";
  if (!key) {
    return res.status(200).json({ ai: { available: false,
      reason: "AI Juice isn't configured — set GEMINI_API_KEY in Vercel." } });
  }
  let d = {};
  try { d = typeof req.body === "string" ? JSON.parse(req.body || "{}") : (req.body || {}); }
  catch (_) { d = {}; }
  if (d.projection == null) {
    return res.status(200).json({ ai: { available: false, reason: "No projection to explain." } });
  }
  const fields = {
    player: d.player, team: d.team, sport: d.sport, stat: d.stat, book_line: d.line,
    model_projection: d.projection, edge_vs_line: d.edge, edge_percent: d.edgePct,
    prob_over: d.probability, floor_p10: d.floor, ceiling_p90: d.ceiling,
    sample_games: d.sample_games, confidence: d.confidence, favored_side: d.side,
  };
  const dataBlock = "DATA (the only facts you may use):\n" +
    Object.entries(fields).filter(([, v]) => v != null).map(([k, v]) => `- ${k}: ${v}`).join("\n");
  const prompt = `${dataBlock}\n\nExplain this projection for a sharp but non-technical reader: ` +
    `why the model landed on ${d.projection} for ${d.player} ${d.stat} (book line ${d.line}), ` +
    `and how confident it is.`;
  const body = {
    systemInstruction: { parts: [{ text: SYSTEM }] },
    contents: [{ role: "user", parts: [{ text: prompt }] }],
    generationConfig: { maxOutputTokens: 600, temperature: 0.4 },
  };
  try {
    const r = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${key}`,
      { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!r.ok) {
      // The API's own error.status + message (e.g. "RESOURCE_EXHAUSTED: You exceeded your
      // current quota...") — a bare HTTP code can't distinguish a spent daily quota from a
      // transient per-minute throttle.
      let detail = String(r.status);
      try {
        const err = ((await r.json()) || {}).error || {};
        if (err.status && err.message) detail = `${err.status}: ${err.message}`;
        else if (err.status) detail = err.status;
      } catch (_) { /* keep the bare status */ }
      return res.status(200).json({ ai: { available: false, reason: `AI request rejected (${detail}).` } });
    }
    const j = await r.json();
    const parts = (((j.candidates || [])[0] || {}).content || {}).parts || [];
    const text = parts.map((p) => p.text || "").join("").trim();
    if (!text) {
      const fb = (j.promptFeedback || {}).blockReason;
      return res.status(200).json({ ai: { available: false,
        reason: fb ? `AI declined (${fb}).` : "AI returned no explanation." } });
    }
    return res.status(200).json({ ai: { available: true, explanation: text, provider: "gemini", model } });
  } catch (e) {
    return res.status(200).json({ ai: { available: false, reason: "AI request failed." } });
  }
}
