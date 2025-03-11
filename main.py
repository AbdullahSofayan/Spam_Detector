import random


class TicTacToe:
    def __init__(self):
        self.board = []

    def create_board(self):
        self.board = [['-' for _ in range(3)] for _ in range(3)]

    def get_random_first_player(self):
        return random.randint(0, 1)

    def fix_spot(self, row, col, player):
        if self.board[row][col] == '-':
            self.board[row][col] = player
            return True
        return False

    def is_player_win(self, player):
        n = len(self.board)
        for i in range(n):
            if all(self.board[i][j] == player for j in range(n)):
                return True
            if all(self.board[j][i] == player for j in range(n)):
                return True
        if all(self.board[i][i] == player for i in range(n)):
            return True
        if all(self.board[i][n - i - 1] == player for i in range(n)):
            return True
        return False

    def is_board_filled(self):
        return all(item != '-' for row in self.board for item in row)

    def swap_player_turn(self, player):
        return 'X' if player == 'O' else 'O'

    def show_board(self):
        for row in self.board:
            print(" ".join(row))
        print()

    def minimax(self, is_maximizing, alpha, beta):
        if self.is_player_win('O'):
            return 1
        if self.is_player_win('X'):
            return -1
        if self.is_board_filled():
            return 0

        if is_maximizing:
            best_score = -float('inf')
            for i in range(3):
                for j in range(3):
                    if self.board[i][j] == '-':
                        self.board[i][j] = 'O'
                        score = self.minimax(False, alpha, beta)
                        self.board[i][j] = '-'
                        best_score = max(score, best_score)
                        alpha = max(alpha, best_score)
                        if beta <= alpha:
                            break
            return best_score
        else:
            best_score = float('inf')
            for i in range(3):
                for j in range(3):
                    if self.board[i][j] == '-':
                        self.board[i][j] = 'X'
                        score = self.minimax(True, alpha, beta)
                        self.board[i][j] = '-'
                        best_score = min(score, best_score)
                        beta = min(beta, best_score)
                        if beta <= alpha:
                            break
            return best_score

    def get_best_move(self):
        best_score = -float('inf')
        best_move = None
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == '-':
                    self.board[i][j] = 'O'
                    score = self.minimax(False, -float('inf'), float('inf'))
                    self.board[i][j] = '-'
                    if score > best_score:
                        best_score = score
                        best_move = (i, j)
        return best_move

    def start(self):
        self.create_board()
        player = 'X' if self.get_random_first_player() == 1 else 'O'

        while True:
            print(f"Player {player}'s turn")
            self.show_board()

            if player == 'X':
                while True:
                    try:
                        row, col = map(int, input("Enter row and column (1-3 each): ").split())
                        if self.fix_spot(row - 1, col - 1, player):
                            break
                        else:
                            print("Invalid move, try again.")
                    except ValueError:
                        print("Enter valid row and column numbers (1-3).")
            else:
                print("Computer is making a move...")
                row, col = self.get_best_move()
                self.fix_spot(row, col, player)

            if self.is_player_win(player):
                self.show_board()
                print(f"Player {player} wins!")
                break
            if self.is_board_filled():
                self.show_board()
                print("It's a draw!")
                break

            player = self.swap_player_turn(player)


if __name__ == "__main__":
    game = TicTacToe()
    game.start()
