package agc017;

import java.util.Scanner;

public class Main {
	public static void main(String[] args){
		Main main = new Main();
		main.solveA();
	}

	static Long[][] pre;
	private void solveA() {
		Scanner sc = new Scanner(System.in);
		int N = sc.nextInt();
		int P = sc.nextInt();
		int zeroNum = 0;
		int oneNum = 0;
		long ans = 0L;
		for (int i = 0; i < N; i++) {
			if (sc.nextInt() % 2 == 0) {
				zeroNum++;
			} else {
				oneNum++;
			}
		}
		pre = new Long[101][];
		for (int i = 0; i < 101; i++) {
			pre[i] = new Long[101];
		}
		for (int i = 0; i <= N; i++) {
			for (int z = 0; z <= i; z++) {
				int o = i - z;
				if (o % 2 == P) {
					ans += comb(zeroNum, z) * comb(oneNum, o);
				}
			}
		}
		System.out.println(ans);
	}

	private long comb(int A, int B) {
		if (pre[A][B] == null) {
			if (A < B) {
				return 0L;
			}
			if (B == 0) {
				return 1L;
			}
			if (A / 2 < B) {
				return comb(A, A - B);
			}
			pre[A][B] = comb(A - 1, B) + comb(A - 1, B - 1);
		}
		return pre[A][B];
	}

	private void solveB() {
		Scanner sc = new Scanner(System.in);
		int N = sc.nextInt();
		long A = sc.nextLong();
		long B = sc.nextLong();
		long C = sc.nextLong();
		long D = sc.nextLong();

		if (judge(N, A, B, C, D)) {
			System.out.println("YES");
		} else {
			System.out.println("NO");
		}
	}

	private boolean judge(int N, long A, long B, long C, long D) {
		for (int minus = 0; minus < N; minus++) {
			long min = -D * minus + C * (N - 1 - minus);
			long max = -C * minus + D * (N - 1 - minus);
			if (min <= B - A && B - A <= max) {
				return true;
			}
		}
		return false;
	}
}