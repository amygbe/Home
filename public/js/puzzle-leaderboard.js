document.addEventListener('DOMContentLoaded', function() {
  var container = document.getElementById('global-leaderboard');
  if (!container) return;

  var APPS_SCRIPT_URL = 'https://script.google.com/macros/s/AKfycbwEI_CavSEg_tYqP0AlMKM3sIBbaU9DG2p5HfF6kdQy9cg1jLM0wBtchk_XxYQnFJycmQ/exec';

  fetch(APPS_SCRIPT_URL + '?mode=global')
    .then(function(response) { return response.json(); })
    .then(function(data) {
      if (!data.success) return;
      renderGlobalLeaderboard(data.leaderboard);
    })
    .catch(function() {});

  function renderGlobalLeaderboard(leaderboard) {
    var tableContainer = document.getElementById('global-leaderboard-table');
    if (!tableContainer) return;

    if (leaderboard.length === 0) {
      tableContainer.innerHTML = '<p class="leaderboard-empty">No solves yet. Be the first!</p>';
      return;
    }

    var html = '<table class="leaderboard-table">';
    html += '<thead><tr>';
    html += '<th>#</th>';
    html += '<th>Name</th>';
    html += '<th>Solved</th>';
    html += '<th>Time</th>';
    html += '<th>Wrong</th>';
    html += '</tr></thead>';
    html += '<tbody>';
    for (var i = 0; i < leaderboard.length; i++) {
      var entry = leaderboard[i];
      html += '<tr>';
      html += '<td>' + (i + 1) + '</td>';
      html += '<td>' + escapeHtml(entry.name) + '</td>';
      html += '<td>' + entry.puzzlesSolved + '/' + entry.totalPuzzles + '</td>';
      html += '<td>' + formatTime(entry.totalSolveTime) + '</td>';
      html += '<td>' + entry.totalWrongGuesses + '</td>';
      html += '</tr>';
    }
    html += '</tbody></table>';
    tableContainer.innerHTML = html;
  }

  function formatTime(totalSeconds) {
    if (!totalSeconds) return '-';
    var mins = Math.floor(totalSeconds / 60);
    var secs = totalSeconds % 60;
    if (mins === 0) return secs + 's';
    return mins + 'm ' + secs + 's';
  }

  function escapeHtml(str) {
    var div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }
});
