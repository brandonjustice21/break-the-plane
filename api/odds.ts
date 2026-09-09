// Break the Plane — Anytime TD odds proxy (OpticOdds v3, OddsJam's successor API)
// Ported from the original Val Town HTTP val (odds_valtown_anytime_td.ts) to a
// Vercel Edge Function living in the same repo/project as the board itself --
// same Fetch-API request/response shape as a Val Town val, so this is a
// near-direct port: only the env-var lookup changed (Deno.env -> process.env)
// and the two lines below were added.
//
// 1) pulls today's NFL fixtures, 2) pulls Anytime TD Scorer odds per fixture from
// all book batches, 3) flattens to
//   { players: { "player name": { price, book, consensus, n, books: [{book,price}] } } }.
// price/book = BEST price across books; consensus = median implied probability
// across books (one price per book); n = number of books quoting the player;
// books = full per-book list. CORS open (harmless now that the board fetches
// this same-origin, but left in case anything else ever calls it cross-origin).
// ?debug=1 returns the raw odds payload for the first fixture batch.
//
// Deploy: this file lives at /api/odds.ts in the break-the-plane repo. Vercel
// serves it at https://break-the-plane.vercel.app/api/odds automatically on
// every push to main -- set the ODDSJAM_KEY environment variable in the
// Vercel project's Settings -> Environment Variables (same key value already
// in use on the Val Town val), then trigger a redeploy so the function picks
// it up.
export const config = { runtime: 'edge' };

const BOOK_BATCHES = [
  ["DraftKings", "FanDuel", "BetMGM", "Caesars", "theScore Bet"],
  ["Fanatics", "Bally Bet", "Hard Rock Bet", "BetRivers", "Bet365"],
  ["Novig", "Kalshi", "PrizePicks", "Underdog", "ProphetX"],
  ["Pinnacle"],
];
const MARKETS = [
  "Anytime Touchdown Scorer",
  "Any Time Touchdown Scorer",
  "To Score A Touchdown",
];
let cache = { at: 0, body: null as string | null };

export default async function (req: Request): Promise<Response> {
  const cors = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, OPTIONS",
    "Content-Type": "application/json",
  };
  if (req.method === "OPTIONS") return new Response(null, { headers: cors });
  const key = process.env.ODDSJAM_KEY;
  if (!key) {
    return new Response(
      JSON.stringify({ error: "ODDSJAM_KEY env var not set" }),
      { status: 500, headers: cors },
    );
  }
  const debug = new URL(req.url).searchParams.get("debug") === "1";
  if (!debug && cache.body && Date.now() - cache.at < 180000) {
    return new Response(cache.body, { headers: cors });
  }
  // Step 1: today's NFL fixtures
  const fxRes = await fetch(
    "https://api.opticodds.com/api/v3/fixtures/active?" +
      new URLSearchParams({ key, sport: "football", league: "nfl" }),
  );
  const fxText = await fxRes.text();
  if (!fxRes.ok) {
    return new Response(
      JSON.stringify({
        error: "fixtures HTTP " + fxRes.status,
        body: fxText.slice(0, 500),
      }),
      { status: 502, headers: cors },
    );
  }
  let fx: any;
  try {
    fx = JSON.parse(fxText);
  } catch {
    return new Response(JSON.stringify({ error: "fixtures non-JSON" }), {
      status: 502,
      headers: cors,
    });
  }
  const fixtures = (fx.data || []).map((f: any) => f.id).filter(Boolean);
  if (!fixtures.length) {
    const body = JSON.stringify({
      updated: Date.now(),
      playerCount: 0,
      players: {},
      note: "no active NFL fixtures",
    });
    return new Response(body, { headers: cors });
  }
  // Step 2: odds per fixture batch (5 fixtures per request) x per book batch (max 5 sportsbooks per request)
  const players: Record<
    string,
    {
      price: number;
      book: string;
      consensus?: number;
      n?: number;
      books?: { book: string; price: number }[];
    }
  > = {};
  const perBook: Record<string, Record<string, number>> = {}; // player → book → best price at that book
  let oddsSeen = 0, firstRaw: string | null = null;
  for (let i = 0; i < fixtures.length; i += 5) {
    const fixtureBatch = fixtures.slice(i, i + 5);
    for (const SPORTSBOOKS of BOOK_BATCHES) {
      const qs = new URLSearchParams({ key });
      fixtureBatch.forEach((id: string) => qs.append("fixture_id", id));
      SPORTSBOOKS.forEach((b) => qs.append("sportsbook", b));
      MARKETS.forEach((m) => qs.append("market", m));
      const oRes = await fetch(
        "https://api.opticodds.com/api/v3/fixtures/odds?" + qs.toString(),
      );
      const oText = await oRes.text();
      if (firstRaw == null) firstRaw = oText;
      if (!oRes.ok) continue;
      let oj: any;
      try {
        oj = JSON.parse(oText);
      } catch {
        continue;
      }
      for (const g of (oj.data || [])) {
        for (const o of (g.odds || [])) {
          oddsSeen++;
          if (!/touchdown/i.test(String(o.market || o.market_id || ""))) {
            continue;
          }
          const line = String(o.selection_line || "").toLowerCase();
          if (line === "no" || line === "under") continue;
          let player = String(o.selection || "").trim();
          if (!player) {
            player = String(o.name || "").replace(
              /\s+(yes|no|over|under)[\s\S]*$/i,
              "",
            ).trim();
          }
          const price = Number(o.price);
          if (!player || !price || Number.isNaN(price)) continue;
          const k = player.toLowerCase().normalize("NFD").replace(
            /\p{Diacritic}/gu,
            "",
          ).replace(/\b(jr|sr|ii|iii|iv)\b/g, "").replace(/[^a-z ]/g, "")
            .replace(
              /\s+/g,
              " ",
            ).trim();
          const book = String(o.sportsbook || "");
          if (!players[k] || price > players[k].price) {
            players[k] = { price, book };
          }
          const pb = perBook[k] = perBook[k] || {};
          if (pb[book] == null || price > pb[book]) pb[book] = price;
        }
      }
    }
  }
  const imp = (a: number) => a > 0 ? 100 / (a + 100) : -a / (-a + 100);
  for (const k of Object.keys(players)) {
    const ps = Object.values(perBook[k] || {}).map(imp).sort((x, y) => x - y);
    if (ps.length) {
      const mid = ps.length % 2
        ? ps[(ps.length - 1) / 2]
        : (ps[ps.length / 2 - 1] + ps[ps.length / 2]) / 2;
      players[k].consensus = Math.round(mid * 10000) / 10000;
      players[k].n = ps.length;
    }
    players[k].books = Object.entries(perBook[k] || {}).map((
      [book, price],
    ) => ({ book, price }));
  }
  if (debug) return new Response(firstRaw || "{}", { headers: cors });
  const body = JSON.stringify({
    updated: Date.now(),
    fixtures: fixtures.length,
    oddsSeen,
    playerCount: Object.keys(players).length,
    players,
  });
  cache = { at: Date.now(), body };
  return new Response(body, { headers: cors });
}
