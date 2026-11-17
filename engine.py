"""Chess engine: legal moves, negamax, alpha-beta, quiescence, opening book.

Perft from the start position is 20, 400, 8902. The book plays e2e4 there.
Board is 0x88. Pieces are 1..6 for white pawn, knight, bishop, rook, queen,
king, and negative for black.
"""

KNIGHT = (-33, -31, -18, -14, 14, 18, 31, 33)
KING = (-17, -16, -15, -1, 1, 15, 16, 17)
BISHOP = (-17, -15, 15, 17)
ROOK = (-16, -1, 1, 16)
VAL = (0, 100, 320, 330, 500, 900, 0)
MATE = 100000
START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
# bit, king from, king to, rook from, squares that must be empty
CASTLE = (
    (1, 4, 6, 7, (5, 6)),
    (2, 4, 2, 0, (1, 2, 3)),
    (4, 116, 118, 119, (117, 118)),
    (8, 116, 114, 112, (113, 114, 115)),
)
OPENING_BOOK = {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -": (20, 52, 0, 0, 0),
}


def on(sq):
    return not sq & 136


def A(b, sq, by):
    """True when side `by` attacks sq. by is +1 (white) or -1 (black)."""
    pawn_d = (-17, -15) if by == 1 else (17, 15)
    for d in pawn_d:
        p = sq + d
        if on(p) and b[p] == by:
            return True
    for d in KNIGHT:
        p = sq + d
        if on(p) and b[p] == by * 2:
            return True
    for d in KING:
        p = sq + d
        if on(p) and b[p] == by * 6:
            return True
    for d in BISHOP:
        p = sq + d
        while on(p):
            x = b[p]
            if x:
                if x == by * 3 or x == by * 5:
                    return True
                break
            p += d
    for d in ROOK:
        p = sq + d
        while on(p):
            x = b[p]
            if x:
                if x == by * 4 or x == by * 5:
                    return True
                break
            p += d
    return False


def Kq(b, side):
    target = side * 6
    for sq, piece in enumerate(b):
        if piece == target:
            return sq
    return -1


def F(fen):
    parts = fen.split()
    b = [0] * 128
    sq = 112
    for ch in parts[0]:
        if ch == "/":
            sq -= 24
        elif ch.isdigit():
            sq += int(ch)
        else:
            kind = "pnbrqk".index(ch.lower()) + 1
            b[sq] = -kind if ch.islower() else kind
            sq += 1
    side = 1 if parts[1] == "w" else -1
    castle = 0
    for i, flag in enumerate("KQkq"):
        if flag in parts[2]:
            castle |= 1 << i
    ep = -1
    if len(parts) > 3 and parts[3] != "-":
        ep = (int(parts[3][1]) - 1) * 16 + ord(parts[3][0]) - 97
    return b, side, castle, ep


def fen_key(b, side, castle, ep):
    rows = []
    for rank in range(7, -1, -1):
        empty = 0
        row = []
        for file in range(8):
            piece = b[rank * 16 + file]
            if piece == 0:
                empty += 1
                continue
            if empty:
                row.append(str(empty))
                empty = 0
            name = "pnbrqkPNBRQK"[abs(piece) - 1 + (6 if piece > 0 else 0)]
            row.append(name)
        if empty:
            row.append(str(empty))
        rows.append("".join(row))
    flags = "".join(ch for bit, ch in ((1, "K"), (2, "Q"), (4, "k"), (8, "q")) if castle & bit)
    ep_s = "-" if ep < 0 else chr(97 + (ep & 7)) + str((ep >> 4) + 1)
    return "/".join(rows) + (" w " if side == 1 else " b ") + (flags or "-") + " " + ep_s


def W(move):
    if not move:
        return "0000"
    fr, to, promo, _ep, _castle = move
    uci = chr(97 + (fr & 7)) + str((fr >> 4) + 1) + chr(97 + (to & 7)) + str((to >> 4) + 1)
    if promo:
        uci += {5: "q", 4: "r", 3: "b", 2: "n"}[promo]
    return uci


def Y(b, move, side, castle, ep):
    """Copy the board and play one move. Returns board, castle, ep."""
    fr, to, promo, is_ep, is_castle = move
    n = b[:]
    piece = n[fr]
    n[fr] = 0
    if is_ep:
        n[to - 16 * side] = 0
    n[to] = side * promo if promo else piece
    if is_castle:
        rook_from = to + 1 if to in (6, 118) else to - 2
        rook_to = to - 1 if to in (6, 118) else to + 1
        n[rook_to] = n[rook_from]
        n[rook_from] = 0
    rights = castle
    if piece == 6:
        rights &= ~3
    elif piece == -6:
        rights &= ~12
    if fr == 7 or to == 7:
        rights &= ~1
    if fr == 0 or to == 0:
        rights &= ~2
    if fr == 119 or to == 119:
        rights &= ~4
    if fr == 112 or to == 112:
        rights &= ~8
    new_ep = (fr + to) // 2 if abs(piece) == 1 and abs(to - fr) == 32 else -1
    return n, rights, new_ep


def _castle_ok(b, side, fr, to, rook_from, empty):
    if b[fr] != side * 6 or b[rook_from] != side * 4:
        return False
    for sq in empty:
        if b[sq]:
            return False
    if A(b, fr, -side):
        return False
    step = 1 if to > fr else -1
    sq = fr + step
    while True:
        if A(b, sq, -side):
            return False
        if sq == to:
            return True
        sq += step


def G(b, side, castle, ep, captures=False):
    """Legal moves. A move is (from, to, promo, ep, castle)."""
    pseudo = []
    enemy = -side
    for sq in range(128):
        if not on(sq):
            continue
        piece = b[sq]
        if piece * side <= 0:
            continue
        kind = abs(piece)
        if kind == 1:
            fwd = 16 * side
            one = sq + fwd
            if on(one) and b[one] == 0:
                rank = one >> 4
                if rank in (0, 7):
                    for promo in (5, 4, 3, 2):
                        pseudo.append((sq, one, promo, 0, 0))
                elif not captures:
                    pseudo.append((sq, one, 0, 0, 0))
                    two = sq + fwd * 2
                    home = 1 if side == 1 else 6
                    if (sq >> 4) == home and b[two] == 0:
                        pseudo.append((sq, two, 0, 0, 0))
            for df in (-1, 1):
                cap = sq + fwd + df
                if not on(cap):
                    continue
                if b[cap] * side < 0:
                    if (cap >> 4) in (0, 7):
                        for promo in (5, 4, 3, 2):
                            pseudo.append((sq, cap, promo, 0, 0))
                    else:
                        pseudo.append((sq, cap, 0, 0, 0))
                elif cap == ep and b[cap] == 0:
                    pseudo.append((sq, cap, 0, 1, 0))
        elif kind in (2, 6):
            for delta in KNIGHT if kind == 2 else KING:
                dest = sq + delta
                if not on(dest) or b[dest] * side > 0:
                    continue
                if captures and b[dest] == 0:
                    continue
                pseudo.append((sq, dest, 0, 0, 0))
            if kind == 6 and not captures:
                for bit, fr, to, rook, empty in CASTLE:
                    white_ok = side == 1 and bit < 4
                    black_ok = side == -1 and bit >= 4
                    if castle & bit and (white_ok or black_ok) and _castle_ok(b, side, fr, to, rook, empty):
                        pseudo.append((fr, to, 0, 0, 1))
        else:
            deltas = BISHOP + ROOK if kind == 5 else BISHOP if kind == 3 else ROOK
            for delta in deltas:
                dest = sq + delta
                while on(dest):
                    if b[dest] == 0:
                        if not captures:
                            pseudo.append((sq, dest, 0, 0, 0))
                        dest += delta
                        continue
                    if b[dest] * side < 0:
                        pseudo.append((sq, dest, 0, 0, 0))
                    break
    legal = []
    for move in pseudo:
        nb, _nc, _ne = Y(b, move, side, castle, ep)
        king = Kq(nb, side)
        if king != -1 and A(nb, king, enemy):
            continue
        legal.append(move)
    return legal


def E(b, side):
    score = 0
    for sq, piece in enumerate(b):
        if not piece or not on(sq):
            continue
        kind = abs(piece)
        rank = sq >> 4
        file = sq & 7
        rel = rank if piece > 0 else 7 - rank
        value = VAL[kind]
        if kind == 1:
            value += rel * 5 + (4 if 2 <= file <= 5 else 0)
        elif kind == 2:
            value += (3 - max(abs(file - 3), abs(file - 4))) * 6 + rel * 2
        elif kind == 3:
            value += rel * 2 + (4 if 2 <= file <= 5 else 0)
        elif kind == 6:
            value -= rel * 3
        score += value if piece > 0 else -value
    return score * side


def _rank_move(b, move):
    fr, to, promo, is_ep, _castle = move
    captured = 1 if is_ep else abs(b[to])
    return promo * 16 + captured * 10 - kind_cost(b[fr])


def kind_cost(piece):
    return VAL[abs(piece)] // 100


def Qs(b, side, castle, ep, alpha, beta, ply, left):
    king = Kq(b, side)
    check = king != -1 and A(b, king, -side)
    if left <= 0:
        return -MATE + ply if check else E(b, side)
    if not check:
        stand = E(b, side)
        if stand >= beta:
            return stand
        if stand > alpha:
            alpha = stand
        best = stand
    else:
        best = -MATE + ply
    moves = G(b, side, castle, ep, captures=not check)
    if check and not moves:
        return -MATE + ply
    moves.sort(key=lambda move: _rank_move(b, move), reverse=True)
    for move in moves:
        nb, nc, ne = Y(b, move, side, castle, ep)
        score = -Qs(nb, -side, nc, ne, -beta, -alpha, ply + 1, left - 1)
        if score > best:
            best = score
        if score > alpha:
            alpha = score
            if alpha >= beta:
                break
    return best


def S(b, side, castle, ep, depth, alpha, beta, ply):
    """Negamax with alpha-beta. Depth 0 drops into quiescence."""
    if depth <= 0:
        return Qs(b, side, castle, ep, alpha, beta, ply, 8)
    moves = G(b, side, castle, ep)
    if not moves:
        king = Kq(b, side)
        if king != -1 and A(b, king, -side):
            return -MATE + ply
        return 0
    moves.sort(key=lambda move: _rank_move(b, move), reverse=True)
    best = -MATE * 2
    for move in moves:
        nb, nc, ne = Y(b, move, side, castle, ep)
        score = -S(nb, -side, nc, ne, depth - 1, -beta, -alpha, ply + 1)
        if score > best:
            best = score
        if score > alpha:
            alpha = score
            if alpha >= beta:
                break
    return best


def Bt(b, side, castle, ep, depth=4):
    """Iterative negamax. The opening book returns e2e4 on the start position."""
    booked = OPENING_BOOK.get(fen_key(b, side, castle, ep))
    if booked:
        return booked, 0
    moves = G(b, side, castle, ep)
    if not moves:
        return None, 0
    best_move = moves[0]
    best_score = 0
    limit = max(1, depth)
    for current in range(1, limit + 1):
        alpha = -MATE * 2
        beta = MATE * 2
        ordered = [best_move] + [move for move in moves if move != best_move]
        ordered.sort(key=lambda move: (move == best_move, _rank_move(b, move)), reverse=True)
        current_move = ordered[0]
        current_score = -MATE * 2
        for move in ordered:
            nb, nc, ne = Y(b, move, side, castle, ep)
            score = -S(nb, -side, nc, ne, current - 1, -beta, -alpha, 1)
            if score > current_score:
                current_score = score
                current_move = move
            if score > alpha:
                alpha = score
        best_move, best_score = current_move, current_score
        if abs(best_score) > MATE - 400:
            break
    return best_move, best_score


def perft(b, side, castle, ep, depth):
    if depth == 0:
        return 1
    total = 0
    for move in G(b, side, castle, ep):
        nb, nc, ne = Y(b, move, side, castle, ep)
        total += perft(nb, -side, nc, ne, depth - 1)
    return total


if __name__ == "__main__":
    board, turn, rights, target = F(START)
    for depth, expected in ((1, 20), (2, 400), (3, 8902)):
        found = perft(board, turn, rights, target, depth)
        print(f"perft {depth}: {found} (harap {expected})")
    move, _score = Bt(board, turn, rights, target, 2)
    print("book:", W(move), "(harap e2e4)")
