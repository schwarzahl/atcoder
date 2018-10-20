package agc015;

import java.util.Scanner;
public class Main {
	public static void main(String[] args){
		Main main = new Main();
		main.solveC();
	}

	private void solveA() {
		Scanner sc = new Scanner(System.in);
		long N = sc.nextLong();
		long A = sc.nextLong();
		long B = sc.nextLong();
		long min = A * (N - 1) + B;
		long max = B * (N - 1) + A;
		if (A > B || (N == 1 && A != B)) {
			System.out.println(0);
		} else {
			System.out.println(max - min + 1);
		}
	}

	private void solveB() {
		Scanner sc = new Scanner(System.in);
		String S = sc.nextLine();
		int N = S.length();
		long sum = 0L;
		for (int i = 0; i < N; i++) {
			if (S.charAt(i) == 'U') {
				sum += 2 * i + 1 * (N - 1 - i);
			} else {
				sum += 1 * i + 2 * (N - 1 - i);
			}
		}
		System.out.println(sum);
	}

	private void solveC() {
		Scanner sc = new Scanner(System.in);
		int N = sc.nextInt();
		int M = sc.nextInt();
		int Q = sc.nextInt();
		sc.nextLine();
		boolean[][] map = new boolean[N + 1][];
		map[0] = new boolean[M + 1];
		for (int r = 1; r <= N; r++) {
			map[r] = new boolean[M + 1];
			String line = sc.nextLine();
			for (int c = 1; c <= M; c++) {
				map[r][c] = line.charAt(c - 1) == '1';
			}
		}
		int[][] sum_v = new int[N + 1][];
		int[][] sum_e_r = new int[N + 1][];
		int[][] sum_e_c = new int[N + 1][];
		sum_v[0] = new int[M + 1];
		sum_e_r[0] = new int[M + 1];
		sum_e_c[0] = new int[M + 1];
		for (int r = 1; r <= N; r++) {
			sum_v[r] = new int[M + 1];
			sum_e_r[r] = new int[M + 1];
			sum_e_c[r] = new int[M + 1];
			for (int c = 1; c <= M; c++) {
				sum_v[r][c] = sum_v[r][c - 1] + sum_v[r - 1][c] - sum_v[r - 1][c - 1] + (map[r][c] ? 1 : 0);
				sum_e_r[r][c] = sum_e_r[r][c - 1] + sum_e_r[r - 1][c] - sum_e_r[r - 1][c - 1] + (map[r][c] && map[r - 1][c] ? 1 : 0);
				sum_e_c[r][c] = sum_e_c[r][c - 1] + sum_e_c[r - 1][c] - sum_e_c[r - 1][c - 1] + (map[r][c] && map[r][c - 1] ? 1 : 0);
			}
		}

		for (int k = 0; k < Q; k++) {
			int x_1 = sc.nextInt();
			int y_1 = sc.nextInt();
			int x_2 = sc.nextInt();
			int y_2 = sc.nextInt();
			int v = sum_v[x_2][y_2] - sum_v[x_1 - 1][y_2] - sum_v[x_2][y_1 - 1] + sum_v[x_1 - 1][y_1 - 1];
			int r = sum_e_r[x_2][y_2] - sum_e_r[x_1][y_2] - sum_e_r[x_2][y_1 - 1] + sum_e_r[x_1][y_1 - 1];
			int c = sum_e_c[x_2][y_2] - sum_e_c[x_1 - 1][y_2] - sum_e_c[x_2][y_1] + sum_e_c[x_1 - 1][y_1];
			System.out.println(v - r - c);
		}
	}
}