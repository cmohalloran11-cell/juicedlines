// Vercel serverless function — today's real AI Juice usage for the signed-in user. Mirrors
// main.py's /api/ai/usage exactly (same table, same limits — see _lib.js). Never increments;
// that only happens inside explain.js/coach.js on an actual call.

const { verifyUser, getUsage, dailyLimit, todayUtc } = require("./_lib");

module.exports = async function handler(req, res) {
  const user = await verifyUser(req);
  if (!user) {
    return res.status(200).json({ signed_in: false });
  }
  const limit = dailyLimit(user.tier);
  const used = await getUsage(user.id, todayUtc());
  return res.status(200).json({
    signed_in: true, used: used == null ? 0 : used, limit, tier: user.tier,
  });
}
