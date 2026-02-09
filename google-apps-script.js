// Google Apps Script for Cryptic Puzzle Leaderboard
//
// SETUP INSTRUCTIONS:
// 1. Add a new column header "solveTime" (column G) after wrongAnswers
// 2. Paste this entire file into the Apps Script editor (replace existing code)
// 3. Deploy → Manage deployments → Edit → Version: New version → Deploy
//
// Sheet headers: timestamp | puzzleId | name | hintsUsed | wrongGuesses | wrongAnswers | solveTime

var SHEET_NAME = 'Solves';
// TOTAL_PUZZLES is computed dynamically from distinct puzzleIds in the sheet

function doPost(e) {
  try {
    var data = JSON.parse(e.postData.contents);
    var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName(SHEET_NAME);

    // Validate required fields
    if (!data.puzzleId || !data.name) {
      return ContentService.createTextOutput(JSON.stringify({
        success: false,
        error: 'Missing required fields'
      })).setMimeType(ContentService.MimeType.JSON);
    }

    // Sanitize name (limit length, strip HTML, normalize case)
    var name = String(data.name).replace(/<[^>]*>/g, '').substring(0, 30).trim().toLowerCase();
    if (!name) {
      return ContentService.createTextOutput(JSON.stringify({
        success: false,
        error: 'Invalid name'
      })).setMimeType(ContentService.MimeType.JSON);
    }

    // Check for duplicate: same name + puzzleId
    var existing = sheet.getDataRange().getValues();
    var puzzleIdStr = String(data.puzzleId);
    for (var i = 1; i < existing.length; i++) {
      if (existing[i][1] === puzzleIdStr && String(existing[i][2]).toLowerCase() === name) {
        return ContentService.createTextOutput(JSON.stringify({
          success: false,
          error: 'duplicate',
          message: 'This name has already been submitted for this puzzle'
        })).setMimeType(ContentService.MimeType.JSON);
      }
    }

    // Append row
    sheet.appendRow([
      new Date().toISOString(),
      puzzleIdStr,
      name,
      parseInt(data.hintsUsed) || 0,
      parseInt(data.wrongGuesses) || 0,
      String(data.wrongAnswers || ''),
      parseInt(data.solveTime) || 0
    ]);

    return ContentService.createTextOutput(JSON.stringify({
      success: true
    })).setMimeType(ContentService.MimeType.JSON);

  } catch (err) {
    return ContentService.createTextOutput(JSON.stringify({
      success: false,
      error: err.message
    })).setMimeType(ContentService.MimeType.JSON);
  }
}

function doGet(e) {
  try {
    var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName(SHEET_NAME);
    var data = sheet.getDataRange().getValues();
    var mode = e.parameter.mode || null;
    var puzzleId = e.parameter.puzzleId || null;

    // Global leaderboard mode
    if (mode === 'global') {
      return handleGlobalLeaderboard(data);
    }

    // Per-puzzle leaderboard (existing behavior)
    var solves = [];
    for (var i = 1; i < data.length; i++) {
      var row = {
        timestamp: data[i][0],
        puzzleId: data[i][1],
        name: data[i][2],
        hintsUsed: data[i][3],
        wrongGuesses: data[i][4],
        wrongAnswers: data[i][5],
        solveTime: data[i][6] || null
      };

      if (!puzzleId || row.puzzleId === puzzleId) {
        solves.push(row);
      }
    }

    // Sort by fewest hints, then fewest wrong guesses
    solves.sort(function(a, b) {
      if (a.hintsUsed !== b.hintsUsed) return a.hintsUsed - b.hintsUsed;
      return a.wrongGuesses - b.wrongGuesses;
    });

    // Collect all wrong answers (without names)
    var wrongAnswersList = [];
    for (var j = 0; j < solves.length; j++) {
      if (solves[j].wrongAnswers) {
        var answers = solves[j].wrongAnswers.split(',');
        for (var k = 0; k < answers.length; k++) {
          var trimmed = answers[k].trim();
          if (trimmed) {
            wrongAnswersList.push(trimmed);
          }
        }
      }
    }

    var result = {
      success: true,
      totalSolves: solves.length,
      leaderboard: solves.map(function(s) {
        return {
          name: s.name,
          hintsUsed: s.hintsUsed,
          wrongGuesses: s.wrongGuesses,
          solveTime: s.solveTime
        };
      }),
      wrongAnswers: wrongAnswersList
    };

    return ContentService.createTextOutput(JSON.stringify(result))
      .setMimeType(ContentService.MimeType.JSON);

  } catch (err) {
    return ContentService.createTextOutput(JSON.stringify({
      success: false,
      error: err.message
    })).setMimeType(ContentService.MimeType.JSON);
  }
}

function handleGlobalLeaderboard(data) {
  // Count distinct puzzleIds and aggregate solves by player name
  var allPuzzleIds = {};
  var players = {};
  for (var i = 1; i < data.length; i++) {
    var name = String(data[i][2]).toLowerCase();
    var puzzleId = data[i][1];
    var wrongGuesses = parseInt(data[i][4]) || 0;
    var solveTime = parseInt(data[i][6]) || 0;

    allPuzzleIds[puzzleId] = true;

    if (!players[name]) {
      players[name] = { puzzles: {}, totalWrongGuesses: 0, totalSolveTime: 0 };
    }

    // Only count first solve per puzzle per player
    if (!players[name].puzzles[puzzleId]) {
      players[name].puzzles[puzzleId] = true;
      players[name].totalWrongGuesses += wrongGuesses;
      players[name].totalSolveTime += solveTime;
    }
  }

  var totalPuzzles = Object.keys(allPuzzleIds).length;

  // Build leaderboard array
  var leaderboard = [];
  for (var playerName in players) {
    var p = players[playerName];
    leaderboard.push({
      name: playerName,
      puzzlesSolved: Object.keys(p.puzzles).length,
      totalPuzzles: totalPuzzles,
      totalWrongGuesses: p.totalWrongGuesses,
      totalSolveTime: p.totalSolveTime
    });
  }

  // Sort: most puzzles solved first, then fastest total time, then fewest wrong guesses
  leaderboard.sort(function(a, b) {
    if (b.puzzlesSolved !== a.puzzlesSolved) return b.puzzlesSolved - a.puzzlesSolved;
    if (a.totalSolveTime !== b.totalSolveTime) return a.totalSolveTime - b.totalSolveTime;
    return a.totalWrongGuesses - b.totalWrongGuesses;
  });

  return ContentService.createTextOutput(JSON.stringify({
    success: true,
    leaderboard: leaderboard
  })).setMimeType(ContentService.MimeType.JSON);
}
