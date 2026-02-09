# Cryptic Puzzle Leaderboard Plan

## Backend: Google Sheets + Google Apps Script

Google Sheets is a good fit here. It's free, requires no server, and you can view/manage all the data directly in a spreadsheet. The alternative would be Firebase/Supabase, but those are overkill for a personal site with a handful of puzzles.

### What you'll need to set up (manual steps):
1. Create a Google Sheet with two sheets/tabs:
   - **Solves** — columns: `timestamp`, `puzzleId`, `name`, `hintsUsed`, `wrongGuesses`, `wrongAnswers` (comma-separated list)
   - **Stats** — auto-calculated stats (or we compute on read)
2. Create a Google Apps Script (Extensions → Apps Script) that handles two endpoints:
   - **POST** — receives a solve submission and appends a row
   - **GET** — returns leaderboard data (all solves for a given puzzle or all puzzles)
3. Deploy the Apps Script as a web app (set to "Anyone" can access)
4. Give me the deployed URL — I'll wire it into the JS

### I will provide:
- The full Apps Script code to paste in
- All the frontend code changes

---

## Frontend Changes

### 1. Track wrong answers during gameplay
- Currently `wrongGuesses` is a counter. We'll also collect the actual wrong answer text into an array as the user plays.

### 2. Name submission in congrats popup
- After solving, the congrats popup will show a text input for the user's name and a "Submit to Leaderboard" button
- Submitting sends: `puzzleId`, `name`, `hintsUsed`, `wrongGuesses`, `wrongAnswers[]`, `timestamp`
- After submit, show a confirmation message
- Name input is optional — users can close without submitting

### 3. Leaderboard section on each puzzle page
Below the puzzle, a leaderboard section loads via the GET endpoint and shows:
- **Total unique solves** count
- **Solver rankings table** sorted by: fewest hints first, then fewest wrong guesses
  - Columns: Rank, Name, Hints Used, Wrong Guesses
- **Wrong answers section** — a list of wrong answers people have submitted (shown without names attached, to keep it fun/anonymous)

### 4. Solve counter update
- The existing "X solved" counter currently uses localStorage (local to each browser). With the backend, it will show the real total from the API instead.

---

## Files Modified

| File | Change |
|------|--------|
| `static/js/cryptic-puzzle.js` | Track wrong answer text, add API calls, render leaderboard, name submission |
| `layouts/shortcodes/cryptic-puzzle.html` | Add name input to congrats popup, add leaderboard container div |
| `static/css/custom.css` | Styles for leaderboard table, name input, wrong answers section |

## New Files

| File | Purpose |
|------|---------|
| (Google Apps Script — not in repo) | I'll provide the code for you to paste into Apps Script |

---

## Summary

- You set up the Google Sheet + Apps Script + deploy it
- I write all the frontend code + provide the Apps Script code
- No ongoing maintenance — data lives in your Google Sheet where you can view/edit/delete entries anytime
