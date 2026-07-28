// Vercel serverless function — reports whether AI Juice is configured (GEMINI_API_KEY set),
// so the static frontend can show the AI on/off state without exposing the key.
module.exports = function handler(req, res) {
  res.status(200).json({
    available: !!process.env.GEMINI_API_KEY,
    provider: "gemini",
    model: process.env.AI_MODEL || "gemini-2.5-flash",
  });
}
