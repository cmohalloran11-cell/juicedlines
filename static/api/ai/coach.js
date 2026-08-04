// Vercel serverless function — grounded AI Coach slate-level strategy recommendation.
// Mirrors ai_juice.py's coach() / _COACH_SYSTEM exactly (this is the static-deploy twin of
// that function, same way explain.js mirrors ai_juice.explain()). It may ONLY use the numbers
// passed in the request body (the dashboard's own coach_context, already computed from real
// board data) — it cannot fabricate a count or percentage. Set GEMINI_API_KEY (and optionally
// AI_MODEL) in Vercel -> Project -> Settings -> Environment Variables.

const SYSTEM = `You are the AI Coach for the Juiced sports-prop research platform. You give a short, actionable STRATEGY recommendation for today's slate as a whole — not a single prop.

Hard rules — follow exactly:
1. Use ONLY the numbers in the DATA block. Never invent a count, percentage, player name, or market condition that is not present there.
2. Never guarantee an outcome or use "lock"/"guaranteed"/"free money" language. This is strategy guidance (how aggressively to play, what to avoid), not a pick.
3. Be concrete and directive: 2-4 short sentences, plain language, aimed at someone deciding how many entries to play and how big. Reference the actual counts from DATA when you make a recommendation (e.g. "only N props clear 80+ Juice today" rather than a vague "a few").
4. If the slate has fewer high-confidence edges than usual (few 80+ Juice props relative to the total pool, or the top entry's projected win% is low), say so plainly and recommend playing smaller/fewer entries. If the slate is strong, say so and explain what makes it strong using the real numbers.`;

module.exports = async function handler(req, res) {
  const key = process.env.GEMINI_API_KEY;
  const model = process.env.AI_MODEL || "gemini-2.5-flash";
  if (!key) {
    return res.status(200).json({ ai: { available: false,
      reason: "AI Coach isn't configured — set GEMINI_API_KEY in Vercel." } });
  }
  let d = {};
  try { d = typeof req.body === "string" ? JSON.parse(req.body || "{}") : (req.body || {}); }
  catch (_) { d = {}; }
  const dataBlock = "DATA (the only facts you may use):\n" +
    Object.entries(d).filter(([, v]) => v != null).map(([k, v]) => `- ${k}: ${v}`).join("\n");
  const prompt = `${dataBlock}\n\nGive today's strategy recommendation.`;
  const body = {
    systemInstruction: { parts: [{ text: SYSTEM }] },
    contents: [{ role: "user", parts: [{ text: prompt }] }],
    generationConfig: { maxOutputTokens: 400, temperature: 0.4 },
  };
  try {
    const r = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${key}`,
      { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!r.ok) {
      return res.status(200).json({ ai: { available: false, reason: `AI request rejected (${r.status}).` } });
    }
    const j = await r.json();
    const parts = (((j.candidates || [])[0] || {}).content || {}).parts || [];
    const text = parts.map((p) => p.text || "").join("").trim();
    if (!text) {
      const fb = (j.promptFeedback || {}).blockReason;
      return res.status(200).json({ ai: { available: false,
        reason: fb ? `AI declined (${fb}).` : "AI returned no recommendation." } });
    }
    return res.status(200).json({ ai: { available: true, strategy: text, provider: "gemini", model } });
  } catch (e) {
    return res.status(200).json({ ai: { available: false, reason: "AI request failed." } });
  }
}
