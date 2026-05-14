(function () {
    // --- state variables ---
    let grid = [];
    let initialGrid = [];
    let rows = 0, cols = 0, moves = 0;
    let timerID = null, startTime = null, isGameOver = false;
    let maxSize = 15;
    let minSize = 1;
    // optimal solution data
    let optimalPressVector = null;        // 1D array (0/1) length rows*cols, or null if unsolvable
    let optimalMoves = null;              // count or null
    let showSolution = false;              // whether to highlight optimal presses

    const boardEl = document.getElementById('game-board');
    const movesEl = document.getElementById('moves');
    const timerEl = document.getElementById('timer');
    const coordsEl = document.getElementById('coords');
    const optimalEl = document.getElementById('optimal');
    const solutionToggleBtn = document.getElementById('solutionToggleBtn');


    function loadDarkMode() {
        const isDark = localStorage.getItem('darkMode') === 'true';
        if (isDark) {
            document.body.classList.add('dark-mode');
        }
    }

    function saveDarkMode() {
        const isDark = document.body.classList.contains('dark-mode');
        localStorage.setItem('darkMode', isDark);
    }

    window.toggleDarkMode = function () {
        document.body.classList.toggle('dark-mode');
        saveDarkMode();                    // ← save immediately
    };

    loadDarkMode();

    // --- core game functions ---
    function initGame(stateArray) {
        initialGrid = stateArray.map(row => [...row]);
        grid = stateArray.map(row => [...row]);
        rows = grid.length;
        cols = grid[0].length;

        moves = 0;
        isGameOver = false;
        movesEl.innerText = `Moves: ${moves}`;
        boardEl.style.gridTemplateColumns = `repeat(${cols}, 50px)`;

        // hide any leftover solution display
        showSolution = false;
        solutionToggleBtn.innerText = '🔍 Show Solution';

        // compute optimal solution for this initial board
        computeAndStoreOptimal(initialGrid);

        renderBoard();
        startTimer();
    }

    // --- compute optimal solution vector and count ---
    // returns { count: number or null, vector: array (0/1) or null }
    function computeOptimalSolution(grid2d) {
        const R = grid2d.length;
        const C = grid2d[0].length;
        const N = R * C;
        if (N === 0) return { count: 0, vector: [] };

        // flatten target (b)
        const b = [];
        for (let r = 0; r < R; r++) {
            for (let c = 0; c < C; c++) {
                b.push(grid2d[r][c]);
            }
        }

        // build coefficient matrix A (N x N) : A[targetIndex][pressIndex] = 1 if press toggles target
        const A = Array.from({ length: N }, () => new Array(N).fill(0));
        for (let r = 0; r < R; r++) {
            for (let c = 0; c < C; c++) {
                const idx = r * C + c;
                const neighbors = [
                    [r, c], [r - 1, c], [r + 1, c], [r, c - 1], [r, c + 1]
                ];
                for (let [nr, nc] of neighbors) {
                    if (nr >= 0 && nr < R && nc >= 0 && nc < C) {
                        const nidx = nr * C + nc;
                        A[nidx][idx] = 1;   // pressing idx toggles cell nidx
                    }
                }
            }
        }

        // augmented matrix [A | b]  (size N x (N+1))
        const aug = [];
        for (let i = 0; i < N; i++) {
            const row = A[i].slice(); // copy coefficients
            row.push(b[i]);            // last element is RHS
            aug.push(row);
        }

        // Gaussian elimination to RREF with pivot tracking
        let row = 0;
        let col = 0;
        const pivotCol = new Array(N).fill(-1); // row -> pivot column

        // forward elimination (row echelon)
        while (row < N && col < N) {
            let sel = -1;
            for (let i = row; i < N; i++) {
                if (aug[i][col] === 1) {
                    sel = i;
                    break;
                }
            }
            if (sel === -1) {
                col++;
                continue;
            }
            [aug[row], aug[sel]] = [aug[sel], aug[row]];
            pivotCol[row] = col;

            for (let i = row + 1; i < N; i++) {
                if (aug[i][col] === 1) {
                    for (let j = col; j <= N; j++) {
                        aug[i][j] ^= aug[row][j];
                    }
                }
            }
            row++;
            col++;
        }

        const rankFinal = row;

        // check for inconsistency: row with all zero coeff but RHS = 1
        for (let i = 0; i < N; i++) {
            let allZero = true;
            for (let j = 0; j < N; j++) {
                if (aug[i][j] !== 0) { allZero = false; break; }
            }
            if (allZero && aug[i][N] === 1) {
                console.warn('System is mathematically inconsistent. No valid solution exists for this board.');
                return { count: null, vector: null }; // unsolvable
            }
        }

        // back substitution to RREF (eliminate above each pivot)
        for (let i = rankFinal - 1; i >= 0; i--) {
            const pc = pivotCol[i];
            for (let j = i - 1; j >= 0; j--) {
                if (aug[j][pc] === 1) {
                    for (let k = pc; k <= N; k++) {
                        aug[j][k] ^= aug[i][k];
                    }
                }
            }
        }

        // collect free variables (columns without pivot)
        const isPivotCol = new Array(N).fill(false);
        for (let i = 0; i < rankFinal; i++) {
            if (pivotCol[i] !== -1) isPivotCol[pivotCol[i]] = true;
        }
        const freeVars = [];
        for (let j = 0; j < N; j++) {
            if (!isPivotCol[j]) freeVars.push(j);
        }
        const freeCount = freeVars.length;

        // particular solution (set free vars = 0)
        // THIS is our immediate, guaranteed valid solution (even if it's not the shortest)
        const x0 = new Array(N).fill(0);
        for (let i = 0; i < rankFinal; i++) {
            const pc = pivotCol[i];
            x0[pc] = aug[i][N];  // RHS after elimination
        }

        // if no free vars, x0 is unique solution
        if (freeCount === 0) {
            const weight = x0.reduce((s, v) => s + v, 0);
            return { count: weight, vector: x0.slice() };
        }

        // null space basis vectors
        const basis = [];
        for (let f of freeVars) {
            const vec = new Array(N).fill(0);
            vec[f] = 1;
            for (let i = 0; i < rankFinal; i++) {
                const pc = pivotCol[i];
                let sum = 0;
                for (let ff of freeVars) {
                    if (aug[i][ff] === 1) sum ^= vec[ff];
                }
                vec[pc] = sum;
            }
            basis.push(vec);
        }

        // ------------------------------------------------------------------
        // FALLBACK LOGIC
        // Lowered threshold from 20 to 10. 
        // 2^10 is 1024 checks (instant). 2^20 is 1 million checks (laggy).
        // If we have too many free variables, just return the valid particular solution!
        // ------------------------------------------------------------------
        const MAX_FREE_VARS_FOR_OPTIMAL = 10;

        console.log(`Free variables: ${freeCount}. ${1 << freeCount} combinations to check for optimality.`);
        if (freeCount > MAX_FREE_VARS_FOR_OPTIMAL) {
            console.log(`Board is large (Free vars: ${freeCount}). Falling back to a guaranteed valid solution to prevent lag.`);
            const weight = x0.reduce((s, v) => s + v, 0);
            return { count: weight, vector: x0.slice() };
        }

        // enumerate all combinations to find the absolute shortest path
        let minWeight = Infinity;
        let bestVector = null;
        const totalCombos = 1 << freeCount;
        for (let mask = 0; mask < totalCombos; mask++) {
            const candidate = x0.slice();
            for (let i = 0; i < freeCount; i++) {
                if (mask >> i & 1) {
                    const bv = basis[i];
                    for (let j = 0; j < N; j++) {
                        candidate[j] ^= bv[j];
                    }
                }
            }
            const w = candidate.reduce((s, v) => s + v, 0);
            if (w < minWeight) {
                minWeight = w;
                bestVector = candidate.slice();
                if (minWeight === 0) break;
            }
        }

        if (bestVector) {
            return { count: minWeight, vector: bestVector };
        } else {
            return { count: null, vector: null };
        }
    }

    function computeAndStoreOptimal(gridState) {
        try {
            const result = computeOptimalSolution(gridState);
            if (result.count === null || result.vector === null) {
                optimalMoves = null;
                optimalPressVector = null;
                optimalEl.innerText = 'Optimal: N/A';
            } else {
                optimalMoves = result.count;
                optimalPressVector = result.vector;  // 1D array of 0/1
                optimalEl.innerText = `Optimal: ${optimalMoves}`;
            }
        } catch (e) {
            console.warn('Optimal calculation failed', e);
            optimalMoves = null;
            optimalPressVector = null;
            optimalEl.innerText = 'Optimal: err';
        }
        // if solution was being shown, hide it because board changed
        if (showSolution) {
            showSolution = false;
            solutionToggleBtn.innerText = '🔍 Show Solution';
        }
    }

    function renderBoard() {
        boardEl.innerHTML = '';
        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                const cell = document.createElement('div');
                cell.classList.add('cell');
                if (grid[r][c] === 1) cell.classList.add('on');

                // optimal solution highlight if active and vector exists
                if (showSolution && optimalPressVector) {
                    const idx = r * cols + c;
                    if (optimalPressVector[idx] === 1) {
                        cell.classList.add('optimal-press');
                    }
                }

                cell.onmouseover = () => {
                    coordsEl.innerText = `Row: ${r}, Col: ${c}`;
                };
                cell.onmouseout = () => {
                    coordsEl.innerText = `Lights Out (Beta)`;
                };
                cell.onclick = () => handleCellClick(r, c);

                boardEl.appendChild(cell);
            }
        }
    }

    function handleCellClick(r, c) {
        if (isGameOver) return;

        const idx = r * cols + c;

        // If in solution mode, check if the clicked piece is part of the solution
        if (showSolution && optimalPressVector) {
            if (optimalPressVector[idx] === 0) {
                alert("Warning: That piece is not part of the optimal solution!");
                return;
            }
        }

        // Keep the optimal solution up-to-date mathematically on every click.
        // Pressing a cell simply toggles its requirement in the optimal vector!
        if (optimalPressVector) {
            optimalPressVector[idx] ^= 1;

            // Adjust the optimal moves count
            if (optimalPressVector[idx] === 1) {
                optimalMoves++; // We added an unnecessary move, so it takes 1 more to solve
            } else {
                optimalMoves--; // We completed a required move
            }
            optimalEl.innerText = `Optimal: ${optimalMoves}`;
        }

        moves++;
        movesEl.innerText = `Moves: ${moves}`;
        toggleNeighbors(r, c);
        renderBoard();
        checkWinCondition();
    }


    function checkWinCondition() {
        const isWon = grid.every(row => row.every(val => val === 0));
        if (isWon) {
            isGameOver = true;
            clearInterval(timerID);
            console.log(`Game won in ${moves} moves. Optimal: ${optimalMoves !== null ? optimalMoves : 'N/A'}`);
            setTimeout(() => alert(`Victory! Moves: ${moves}  (Optimal: ${optimalMoves !== null ? optimalMoves : 'N/A'})`), 50);
        }
    }

    function startTimer() {
        if (timerID) clearInterval(timerID);
        startTime = Date.now();
        timerID = setInterval(() => {
            const totalSec = Math.floor((Date.now() - startTime) / 1000);
            timerEl.innerText = `Time: ${Math.floor(totalSec / 60)}:${(totalSec % 60).toString().padStart(2, '0')}`;
        }, 1000);
    }

    function toggleNeighbors(r, c) {
        toggleFn(grid, r, c);
        toggleFn(grid, r - 1, c);
        toggleFn(grid, r + 1, c);
        toggleFn(grid, r, c - 1);
        toggleFn(grid, r, c + 1);
    }

    const toggleFn = (yourGrid, row, col) => {
        if (row >= 0 && row < rows && col >= 0 && col < cols) {
            yourGrid[row][col] ^= 1;
        }
    };


    window.genRandomGrid = genRandomGrid; // expose for button onclick

    function genRandomGrid(solvable) {
        console.log(`Generating ${solvable ? 'solvable' : 'random'} grid...`);
        let size = parseInt(document.getElementById('grid-size').value) || 5;
        if (size < minSize) size = minSize;
        if (size > maxSize) size = maxSize;
        rows = size; cols = size;

        var newGrid;
        if (solvable) {
            newGrid = Array.from({ length: rows }, () => Array(cols).fill(0));
            const numRandomPresses = rows * cols * 2; // heuristic: more presses for larger boards
            for (let i = 0; i < numRandomPresses; i++) {
                const r = Math.floor(Math.random() * rows);
                const c = Math.floor(Math.random() * cols);
                toggleFn(newGrid, r, c);
            }
        }
        else {
            newGrid = Array.from({ length: rows }, () =>
                Array.from({ length: cols }, () => Math.random() < 0.5 ? 1 : 0)
            );
        }
        initGame(newGrid);
    }

    window.loadCustom = function () {
        const raw = document.getElementById('custom-input').value.trim();
        if (!raw) return;

        let newGrid;
        try {
            // Parse "JS/Python-like" list-of-lists safely
            // Supports: [[1,0],[0,1]] and also with spaces
            const parsed = JSON.parse(raw.replace(/'/g, '"')); // optional support for single quotes
            // basic validation
            if (!Array.isArray(parsed) || !Array.isArray(parsed[0])) {
                throw new Error("Not a 2D array");
            }
            newGrid = parsed.map(row => row.map(v => (v === 1 || v === '1') ? 1 : 0));
        } catch (e) {
            alert("Invalid input. Use a 2D array like: [[1,0],[0,1]]");
            return;
        }

        const h = newGrid.length;
        const w = newGrid[0].length;

        if (h === 0 || w === 0) {
            alert("Grid cannot be empty.");
            return;
        }
        if (!newGrid.every(row => row.length === w)) {
            alert("All rows must have same length");
            return;
        }

        if (h > maxSize || w > maxSize) {
            alert(`Grid too large (max ${maxSize}x${maxSize}). Optimal may be slow.`);
        }

        initGame(newGrid);
    };


    window.resetCurrent = function () {
        initGame(initialGrid);
    };

    // Toggle optimal solution highlighting
    window.toggleOptimalSolution = function () {
        if (!optimalPressVector) {
            alert("No optimal solution available for this board.");
            return;
        }
        showSolution = !showSolution;
        solutionToggleBtn.innerText = showSolution ? '🙈 Hide Solution' : '🔍 Show Solution';
        renderBoard(); // re-render with/without class
    };

    // start with default solvable random
    genRandomGrid(true);

})();
