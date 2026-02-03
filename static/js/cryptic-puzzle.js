document.addEventListener('DOMContentLoaded', function() {
  var puzzle = document.querySelector('.cryptic-puzzle');
  if (!puzzle) return;

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

  var hintsUsed = 0;
  var wrongGuesses = 0;
  var totalHints = document.querySelectorAll('.hint-option').length;

  // Solved counter using localStorage
  function getSolvedCount() {
    var counts = JSON.parse(localStorage.getItem('puzzleSolvedCounts') || '{}');
    return counts[puzzleId] || 0;
  }

  function incrementSolvedCount() {
    var counts = JSON.parse(localStorage.getItem('puzzleSolvedCounts') || '{}');
    counts[puzzleId] = (counts[puzzleId] || 0) + 1;
    localStorage.setItem('puzzleSolvedCounts', JSON.stringify(counts));
    return counts[puzzleId];
  }

  // Display initial solved count
  if (solvedCountEl) {
    solvedCountEl.textContent = getSolvedCount();
  }

  var wrongMessages = [
    "Try again!",
    "Not quite...",
    "Keep trying!",
    "Almost there?",
    "Better luck next time!"
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
    // Fire the confetti!
    fireConfetti();

    // Increment and update solved count
    var newCount = incrementSolvedCount();
    if (solvedCountEl) {
      solvedCountEl.textContent = newCount;
    }

    congratsPopup.style.display = 'flex';

    // Update congrats text based on hints and wrong guesses
    var congratsText = document.getElementById('congrats-text');
    var hintText = hintsUsed === 0 ? 'no hints' : (hintsUsed === 1 ? '1 hint' : hintsUsed + ' hints');
    var guessText = wrongGuesses === 0 ? 'no wrong guesses' : (wrongGuesses === 1 ? '1 wrong guess' : wrongGuesses + ' wrong guesses');
    congratsText.textContent = "Solved with " + hintText + " and " + guessText + "!";

    letterBoxes.forEach(function(box) {
      box.disabled = true;
      box.classList.add('correct');
    });
    if (checkBtn) checkBtn.disabled = true;
    if (hintsBtn) hintsBtn.disabled = true;
  }

  // Fallback copy function for mobile
  function copyToClipboard(text) {
    // Try modern clipboard API first
    if (navigator.clipboard && navigator.clipboard.writeText) {
      return navigator.clipboard.writeText(text).then(function() {
        return true;
      }).catch(function() {
        return fallbackCopy(text);
      });
    }
    return Promise.resolve(fallbackCopy(text));
  }

  function fallbackCopy(text) {
    var textArea = document.createElement('textarea');
    textArea.value = text;
    textArea.style.position = 'fixed';
    textArea.style.left = '-9999px';
    textArea.style.top = '0';
    textArea.setAttribute('readonly', '');
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();
    try {
      document.execCommand('copy');
      document.body.removeChild(textArea);
      return true;
    } catch (err) {
      document.body.removeChild(textArea);
      return false;
    }
  }

  // Share result
  if (shareBtn) {
    shareBtn.addEventListener('click', function() {
      var puzzleTitle = congratsPopup.dataset.puzzleTitle || 'Cryptic Puzzle';
      // Convert "Cryptic Puzzle #1" to "Amy's Cryptic #1"
      var shareTitle = puzzleTitle.replace('Cryptic Puzzle', "Amy's Cryptic");

      var hintText = hintsUsed === 0 ? 'no hints' : (hintsUsed === 1 ? '1 hint' : hintsUsed + ' hints');
      var guessText = wrongGuesses === 0 ? 'no wrong guesses' : (wrongGuesses === 1 ? '1 wrong guess' : wrongGuesses + ' wrong guesses');

      var shareText = shareTitle + '\n';
      shareText += 'Solved with ' + hintText + ' and ' + guessText + '.\n';
      shareText += 'Also please refer Amy for a job!!!\n\n';
      shareText += window.location.href;

      copyToClipboard(shareText).then(function(success) {
        if (success) {
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
