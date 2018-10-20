package abc042;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;
public class Main {
	static final long MOD = 1000000007L;

	public static void main(String[] args){
		Main main = new Main();
		main.solveD();
	}

	private void solveA() {
		Scanner sc = new Scanner(System.in);
		int five = 0;
		int seven = 0;
		for (int i = 0; i < 3; i++) {
			switch (sc.nextInt()) {
			case 5:
				five++;
				break;
			case 7:
				seven++;
				break;
			default:
			}
		}
		if (five == 2 && seven == 1) {
			System.out.println("YES");
		} else {
			System.out.println("NO");
		}
	}

	private void solveB() {
		Scanner sc = new Scanner(System.in);
		int N = sc.nextInt();
		int L = sc.nextInt();
		List<String> list = new ArrayList<String>();
		for (int i = 0; i < N; i++) {
			list.add(sc.next());
		}
		Collections.sort(list);
		for (String str : list) {
			System.out.print(str);
		}
		System.out.println();
	}

	private void solveC() {
		Scanner sc = new Scanner(System.in);
		String N = sc.next();
		int K = sc.nextInt();
		boolean[] forbid = new boolean[10];
		int[] next = new int[10];
		for (int i = 0; i < 10; i++) {
			forbid[i] = false;
			next[i] = i;
		}
		for (int i = 0; i < K; i++) {
			forbid[sc.nextInt()] = true;
		}
		for (int i = 0; i < 10; i++) {
			for (int j = 0; j < 10; j++) {
				if (!forbid[(i + j) % 10]) {
					next[i] = i + j;
					break;
				}
			}
		}

		int L = N.length();
		String answer = "";
		boolean lift = false;
		for (int i = 0; i < N.length(); i++) {
			char c = N.charAt(L - i - 1);
			int ctoi = c - '0' + (lift ? 1 : 0);
			int ni = next[ctoi % 10] % 10;
			if (ctoi == ni) {
				answer = ni + answer;
			} else {
				int AL = answer.length();
				answer = "" + ni;
				for (int j = 0; j < AL; j++) {
					answer = answer + next[0];
				}
			}
			lift = (ctoi >= 10 || next[ctoi] >= 10);
		}
		if (lift) {
			System.out.print(next[1]);
		}
		System.out.println(answer);
	}

	private void solveD() {
		Scanner sc = new Scanner(System.in);
		long H = sc.nextLong();
		long W = sc.nextLong();
		long A = sc.nextLong();
		long B = sc.nextLong();

		long startRowIndex = H - A;
		long startColIndex = B + 1L;
		long answer = 0L;
		for (long shiftIndex = 0L; startRowIndex - shiftIndex > 0L && startColIndex + shiftIndex <= W; shiftIndex++) {
			answer += pointComb(H, W, H - A - shiftIndex, B + 1L + shiftIndex);
			answer %= MOD;
		}
		System.out.println(answer);
	}

	/**
	 * h行w列の格子について点(r, c)を通る組み合わせを求めます
	 * @param h 格子の行数
	 * @param w 格子の列数
	 * @param r 通る点の行index
	 * @param c 通る点の列index
	 * @return C(r + c - 2, c - 1) * C(h - r + w - c, w - c)
	 */
	private long pointComb(long h, long w, long r, long c) {
		if (r < 1 || c < 1 || r > h || c > w) return 1;
		return combination(r + c - 2L, c - 1L) * combination(h - r + w - c, w - c);
	}

	/**
	 * n個からm個選ぶ組み合わせの数を求めます
	 * @param n 全体の数
	 * @param m 選ぶ数
	 * @return C(n, m)
	 */
	private long combination(long n, long m) {
		if (n / 2L < m) {
			return combination(n , n - m);
		}
		if (n <= 0 || m <= 0) {
			return 1L;
		}
		return (combination(n, m - 1) * (n - m + 1) / m) % MOD;
	}
}