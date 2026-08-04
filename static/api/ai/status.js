// Vercel serverless function — reports whether AI Juice is configured (GEMINI_API_KEY set),
// so the static frontend can show the AI on/off state without exposing the key.
module.exports = function handler(req, res) {
  const available = !!process.env.GEMINI_API_KEY;
  res.status(200).json({
    available,
    provider: "gemini",
    model: process.env.AI_MODEL || "gemini-2.5-flash",
    // Free-tier cap for gemini-2.5-flash, measured live from the API's own 429 error body
    // (2026-08-04): "generate_content_free_tier_requests, limit: 20". Update if Google
    // changes it; goes away once billing is enabled on the key (no cap left to show then).
    rpm_limit: available ? 20 : null,
  });
}
