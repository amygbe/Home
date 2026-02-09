document.addEventListener('DOMContentLoaded', function() {
  var puzzle = document.querySelector('.cryptic-puzzle');
  if (!puzzle) return;

                         
  var APPS_SCRIPT_URL = 'https://script.google.com/macros/s/AKfycbzLIpdRXnenSFfs0vqY0-Cp2ILhq113IpsSJsPQCIKFmI8GsUTIsZq5pGJgX-chb3QMkg/exec';

  var answerHash = puzzle.dataset.answerHash;
  var answerLength = parseInt(puzzle.dataset.answerLength) || 0;
  var puzzleId = puzzle.dataset.puzzleId;

  var letterBoxes = document.querySelectorAll('.letter-box');
  var checkBtn = document.getElementById('check-answer-btn');
  var hintsBtn = document.getElementById('hints-btn');
  var hintsPanel = document.getElementById('hints-panel');
  var floatingMsg = document.getElementById('floating-message');
  var congratsPopup = document.getElementById('congrats-popup');
  var closeCongrats = document.getElementById('close-congrats');
  var shareBtn = document.getElementById('share-result');
  var shareCopied = document.getElementById('share-copied');
  var solvedCountEl = document.getElementById('solved-count');
  var submitBtn = document.getElementById('submit-leaderboard');
  var solverNameInput = document.getElementById('solver-name');
  var submitStatus = document.getElementById('submit-status');
  var nameSection = document.getElementById('name-section');
  var totalSolvesEl = document.getElementById('total-solves');

  var hintsUsed = 0;
  var wrongGuesses = 0;
  var wrongAnswersList = [];
  var solveTimeSeconds = 0;
  var puzzleStartTime = Date.now();
  var totalHints = document.querySelectorAll('.hint-option').length;

  // Load leaderboard on page load
  fetchLeaderboard();

  var wrongMessages = [
    "Try again!",
    "Not quite!",
    "Keep trying!",
    "Almost there!",
    "Better luck next time!",
    "I believe in you!",
    "How did you even get that?",
    "not sure about that one...",
    "What lol",
    "cmonnnnn",
    "It's not that hard",
  ];

  // Letter box navigation
  letterBoxes.forEach(function(box, index) {
    box.addEventListener('input', function(e) {
      var value = e.target.value.toUpperCase();
      e.target.value = value;
      if (value.length === 1 && index < letterBoxes.length - 1) {
        letterBoxes[index + 1].focus();
      }
    });

    box.addEventListener('keydown', function(e) {
      if (e.key === 'Backspace' && e.target.value === '' && index > 0) {
        letterBoxes[index - 1].focus();
        letterBoxes[index - 1].value = '';
        e.preventDefault();
      }
      if (e.key === 'ArrowLeft' && index > 0) {
        letterBoxes[index - 1].focus();
        e.preventDefault();
      }
      if (e.key === 'ArrowRight' && index < letterBoxes.length - 1) {
        letterBoxes[index + 1].focus();
        e.preventDefault();
      }
      if (e.key === 'Enter') {
        checkAnswer();
      }
    });

    box.addEventListener('focus', function() {
      this.select();
    });
  });

  // Simple SHA-256 fallback for non-secure contexts (mobile HTTP)
  function sha256Fallback(str) {
    function rightRotate(value, amount) {
      return (value >>> amount) | (value << (32 - amount));
    }
    var mathPow = Math.pow;
    var maxWord = mathPow(2, 32);
    var lengthProperty = 'length';
    var i, j;
    var result = '';
    var words = [];
    var asciiBitLength = str[lengthProperty] * 8;
    var hash = [];
    var k = [];
    var primeCounter = 0;
    var isComposite = {};
    for (var candidate = 2; primeCounter < 64; candidate++) {
      if (!isComposite[candidate]) {
        for (i = 0; i < 313; i += candidate) {
          isComposite[i] = candidate;
        }
        hash[primeCounter] = (mathPow(candidate, .5) * maxWord) | 0;
        k[primeCounter++] = (mathPow(candidate, 1 / 3) * maxWord) | 0;
      }
    }
    str += '\x80';
    while (str[lengthProperty] % 64 - 56) str += '\x00';
    for (i = 0; i < str[lengthProperty]; i++) {
      j = str.charCodeAt(i);
      if (j >> 8) return;
      words[i >> 2] |= j << ((3 - i) % 4) * 8;
    }
    words[words[lengthProperty]] = ((asciiBitLength / maxWord) | 0);
    words[words[lengthProperty]] = (asciiBitLength);
    for (j = 0; j < words[lengthProperty];) {
      var w = words.slice(j, j += 16);
      var oldHash = hash;
      hash = hash.slice(0, 8);
      for (i = 0; i < 64; i++) {
        var w15 = w[i - 15], w2 = w[i - 2];
        var a = hash[0], e = hash[4];
        var temp1 = hash[7]
          + (rightRotate(e, 6) ^ rightRotate(e, 11) ^ rightRotate(e, 25))
          + ((e & hash[5]) ^ ((~e) & hash[6]))
          + k[i]
          + (w[i] = (i < 16) ? w[i] : (
            w[i - 16]
            + (rightRotate(w15, 7) ^ rightRotate(w15, 18) ^ (w15 >>> 3))
            + w[i - 7]
            + (rightRotate(w2, 17) ^ rightRotate(w2, 19) ^ (w2 >>> 10))
          ) | 0);
        var temp2 = (rightRotate(a, 2) ^ rightRotate(a, 13) ^ rightRotate(a, 22))
          + ((a & hash[1]) ^ (a & hash[2]) ^ (hash[1] & hash[2]));
        hash = [(temp1 + temp2) | 0].concat(hash);
        hash[4] = (hash[4] + temp1) | 0;
      }
      for (i = 0; i < 8; i++) {
        hash[i] = (hash[i] + oldHash[i]) | 0;
      }
    }
    for (i = 0; i < 8; i++) {
      for (j = 3; j + 1; j--) {
        var b = (hash[i] >> (j * 8)) & 255;
        result += ((b < 16) ? 0 : '') + b.toString(16);
      }
    }
    return result;
  }

  // Hash function with fallback
  async function hashAnswer(input) {
    var normalized = input.toUpperCase().trim().replace(/\s/g, '');

    // Try native crypto first (requires HTTPS)
    if (window.crypto && window.crypto.subtle) {
      try {
        var encoder = new TextEncoder();
        var data = encoder.encode(normalized);
        var hashBuffer = await crypto.subtle.digest('SHA-256', data);
        var hashArray = Array.from(new Uint8Array(hashBuffer));
        return hashArray.map(function(b) {
          return b.toString(16).padStart(2, '0');
        }).join('');
      } catch (e) {
        // Fall through to fallback
      }
    }

    // Fallback for HTTP/mobile
    return sha256Fallback(normalized);
  }

  // Get answer from boxes
  function getAnswer() {
    var answer = '';
    letterBoxes.forEach(function(box) {
      answer += box.value || '';
    });
    return answer;
  }

  // Check answer
  async function checkAnswer() {
    var userInput = getAnswer();
    if (userInput.length < answerLength) {
      showFloatingMessage('Fill in all letters!');
      return;
    }

    var userHash = await hashAnswer(userInput);

    if (userHash === answerHash) {
      showCongrats();
    } else {
      wrongGuesses++;
      wrongAnswersList.push(userInput.toUpperCase());
      var msg = wrongMessages[Math.floor(Math.random() * wrongMessages.length)];
      showFloatingMessage(msg);
      // Shake boxes
      letterBoxes.forEach(function(box) {
        box.classList.add('shake');
        setTimeout(function() {
          box.classList.remove('shake');
        }, 500);
      });
    }
  }

  // Floating message (fades down)
  function showFloatingMessage(msg) {
    floatingMsg.textContent = msg;
    floatingMsg.classList.remove('fade-down');
    void floatingMsg.offsetWidth; // Trigger reflow
    floatingMsg.classList.add('fade-down');
  }

  // Confetti explosion
  function fireConfetti() {
    var duration = 3000;
    var end = Date.now() + duration;

    // Initial big burst
    confetti({
      particleCount: 150,
      spread: 180,
      origin: { y: 0.6 }
    });

    // Continuous confetti
    var interval = setInterval(function() {
      if (Date.now() > end) {
        clearInterval(interval);
        return;
      }

      // Left side
      confetti({
        particleCount: 50,
        angle: 60,
        spread: 80,
        origin: { x: 0, y: 0.6 }
      });

      // Right side
      confetti({
        particleCount: 50,
        angle: 120,
        spread: 80,
        origin: { x: 1, y: 0.6 }
      });
    }, 250);
  }

  // Congrats popup
  function showCongrats() {
    fireConfetti();

    solveTimeSeconds = Math.round((Date.now() - puzzleStartTime) / 1000);

    congratsPopup.style.display = 'flex';

    var congratsText = document.getElementById('congrats-text');
    var hintText = hintsUsed === 0 ? 'no hints' : (hintsUsed === 1 ? '1 hint' : hintsUsed + ' hints');
    var guessText = wrongGuesses === 0 ? 'no wrong guesses' : (wrongGuesses === 1 ? '1 wrong guess' : wrongGuesses + ' wrong guesses');
    var timeText = formatSolveTime(solveTimeSeconds);
    congratsText.textContent = "Solved in " + timeText + " with " + hintText + " and " + guessText + "!";

    letterBoxes.forEach(function(box) {
      box.disabled = true;
      box.classList.add('correct');
    });
    if (checkBtn) checkBtn.disabled = true;
    if (hintsBtn) hintsBtn.disabled = true;
  }

  // Submit to leaderboard
  if (submitBtn) {
    submitBtn.addEventListener('click', function() {
      var name = solverNameInput.value.trim();
      if (!name) {
        solverNameInput.focus();
        solverNameInput.style.borderColor = '#c0392b';
        return;
      }

      if (!APPS_SCRIPT_URL) {
        submitStatus.textContent = 'Leaderboard not configured yet';
        submitStatus.style.display = 'block';
        return;
      }

      submitBtn.disabled = true;
      submitBtn.textContent = 'Submitting...';

      fetch(APPS_SCRIPT_URL, {
        method: 'POST',
        body: JSON.stringify({
          puzzleId: puzzleId,
          name: name,
          hintsUsed: hintsUsed,
          wrongGuesses: wrongGuesses,
          wrongAnswers: wrongAnswersList.join(', '),
          solveTime: solveTimeSeconds
        })
      })
      .then(function(response) { return response.json(); })
      .then(function(result) {
        if (result.success) {
          nameSection.innerHTML = '<p class="submit-status" style="display:block; color:#4a7c59;">Submitted! You\'re on the leaderboard.</p>';
          fetchLeaderboard();
        } else if (result.error === 'duplicate') {
          nameSection.innerHTML = '<p class="submit-status" style="display:block; color:#888;">This name has already been submitted for this puzzle.</p>';
        } else {
          submitStatus.textContent = 'Something went wrong. Try again!';
          submitStatus.style.display = 'block';
          submitBtn.disabled = false;
          submitBtn.textContent = 'Submit to Leaderboard';
        }
      })
      .catch(function() {
        submitStatus.textContent = 'Network error. Try again!';
        submitStatus.style.display = 'block';
        submitBtn.disabled = false;
        submitBtn.textContent = 'Submit to Leaderboard';
      });
    });
  }

  // Reset name input border on typing
  if (solverNameInput) {
    solverNameInput.addEventListener('input', function() {
      solverNameInput.style.borderColor = '';
    });
  }

  // Fetch and render leaderboard
  function fetchLeaderboard() {
    if (!APPS_SCRIPT_URL) return;

    var url = APPS_SCRIPT_URL + '?puzzleId=' + encodeURIComponent(puzzleId);
    fetch(url)
      .then(function(response) { return response.json(); })
      .then(function(data) {
        if (!data.success) return;

        // Update total solves
        if (totalSolvesEl) {
          totalSolvesEl.textContent = data.totalSolves;
        }
        if (solvedCountEl) {
          solvedCountEl.textContent = data.totalSolves;
        }

        // Render leaderboard table
        var container = document.getElementById('leaderboard-table-container');
        if (container && data.leaderboard.length > 0) {
          var html = '<table class="leaderboard-table">';
          html += '<thead><tr><th>#</th><th>Name</th><th>Hints</th><th>Wrong</th><th>Time</th></tr></thead>';
          html += '<tbody>';
          for (var i = 0; i < data.leaderboard.length; i++) {
            var entry = data.leaderboard[i];
            html += '<tr>';
            html += '<td>' + (i + 1) + '</td>';
            html += '<td>' + escapeHtml(entry.name) + '</td>';
            html += '<td>' + entry.hintsUsed + '</td>';
            html += '<td>' + entry.wrongGuesses + '</td>';
            html += '<td>' + formatSolveTime(entry.solveTime) + '</td>';
            html += '</tr>';
          }
          html += '</tbody></table>';
          container.innerHTML = html;
        } else if (container) {
          container.innerHTML = '<p class="leaderboard-empty">No solves yet. Be the first!</p>';
        }

      })
      .catch(function() {
        // Silently fail - leaderboard is non-essential
      });
  }

  function escapeHtml(str) {
    var div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  function formatSolveTime(totalSeconds) {
    if (!totalSeconds) return '-';
    var mins = Math.floor(totalSeconds / 60);
    var secs = totalSeconds % 60;
    if (mins === 0) return secs + 's';
    return mins + 'm ' + secs + 's';
  }

  // Share or copy text - uses native share on mobile, clipboard on desktop
  function shareOrCopy(text, title, callback) {
    var isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);

    // Use native share API only on mobile
    if (isMobile && navigator.share) {
      navigator.share({
        title: title,
        text: text
      }).then(function() {
        callback(true, 'shared');
      }).catch(function(err) {
        // User cancelled or error - don't show error for cancel
        if (err.name !== 'AbortError') {
          callback(false);
        }
      });
      return;
    }

    // Use clipboard on desktop
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(function() {
        callback(true, 'copied');
      }).catch(function() {
        callback(false);
      });
    } else {
      // Last resort: execCommand
      var textArea = document.createElement('textarea');
      textArea.value = text;
      textArea.style.cssText = 'position:fixed;top:0;left:0;opacity:0;';
      document.body.appendChild(textArea);
      textArea.focus();
      textArea.select();
      var success = false;
      try {
        success = document.execCommand('copy');
      } catch (err) {
        success = false;
      }
      document.body.removeChild(textArea);
      callback(success, 'copied');
    }
  }

  // Share result
  if (shareBtn) {
    shareBtn.addEventListener('click', function() {
      var puzzleTitle = congratsPopup.dataset.puzzleTitle || "Amy's Cryptic";

      var hintText = hintsUsed === 0 ? 'no hints' : (hintsUsed === 1 ? '1 hint' : hintsUsed + ' hints');
      var guessText = wrongGuesses === 0 ? 'no wrong guesses' : (wrongGuesses === 1 ? '1 wrong guess' : wrongGuesses + ' wrong guesses');

      var timeShareText = solveTimeSeconds > 0 ? ' in ' + formatSolveTime(solveTimeSeconds) : '';
      var shareText = puzzleTitle + '\n';
      shareText += 'Solved' + timeShareText + ' with ' + hintText + ' and ' + guessText + '.\n';
      shareText += 'wow that\'s really amazing!\n\n';
      shareText += window.location.href;

      shareOrCopy(shareText, puzzleTitle, function(success, method) {
        if (success && method === 'copied') {
          shareCopied.style.display = 'block';
          setTimeout(function() {
            shareCopied.style.display = 'none';
          }, 2000);
        }
      });
    });
  }

  // Close congrats
  if (closeCongrats) {
    closeCongrats.addEventListener('click', function() {
      congratsPopup.style.display = 'none';
    });
  }

  // Check button
  if (checkBtn) {
    checkBtn.addEventListener('click', function(e) {
      e.preventDefault();
      checkAnswer();
    });
  }

  // Hints button - toggle panel
  function toggleHints() {
    if (hintsPanel.style.display === 'none') {
      hintsPanel.style.display = 'block';
      hintsPanel.classList.add('slide-in');
    } else {
      hintsPanel.style.display = 'none';
    }
  }

  if (hintsBtn && hintsPanel) {
    hintsBtn.addEventListener('click', function(e) {
      e.preventDefault();
      toggleHints();
    });
  }

  // Hint options - click to reveal
  var hintOptions = document.querySelectorAll('.hint-option');
  hintOptions.forEach(function(option) {
    option.addEventListener('click', function() {
      var content = option.querySelector('.hint-content');
      var icon = option.querySelector('.hint-icon');
      if (content.style.display === 'none') {
        content.style.display = 'block';
        icon.textContent = '●';
        option.classList.add('revealed');
        hintsUsed++;
      }
    });
  });
});
