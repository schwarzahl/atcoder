package arc079;

import java.util.Scanner;

public class Main {
	public static void main(String[] args) {
		Main main = new Main();
		main.solveE();
	}

	private void solveE() {
		Scanner sc = new Scanner(System.in);
		int N = sc.nextInt();
		long[] a = new long[N];
		long answer = 0L;
		for (int i = 0; i < N; i++) {
			a[i] = sc.nextLong();
		}
		while(true) {
			long plus = 0L;
			for (int i = 0; i < N; i++) {
				long div = a[i] / N;
				plus += div;
				a[i] = (a[i] % N) - div;
			}
			if (plus == 0) {
				break;
			} else {
				answer += plus;
				for (int i = 0; i < N; i++) {
					a[i] += plus;
				}
			}
		}
		System.out.println(answer);
	}

	private void solveD() {
		Scanner sc = new Scanner(System.in);
		long K = sc.nextLong();
		long M = K / 50L;
		long N = K % 50L;
		if (M == 0) {
			if (N == 0) {
				System.out.println(2);
				System.out.println("1 1");
			} else if (N == 1) {
				System.out.println(2);
				System.out.println("0 2");
			} else {
				System.out.println(N);
				System.out.print(N);
				for (int i = 1; i < N; i++) {
					System.out.print(" " + N);
				}
				System.out.println();
			}
		} else {
			System.out.println(50);
			System.out.print(49L + M - N);
			int i = 1;
			for (; i < 50 - N; i++) {
				System.out.print(" " + (49L + M - N));
			}
			for (; i < 50; i++) {
				System.out.print(" " + (100L + M - N));
			}
		}
	}

	private void solveC() {
		Scanner sc = new Scanner(System.in);
		int N = sc.nextInt();
		int M = sc.nextInt();
		boolean[] a = new boolean[N + 1];
		boolean[] b = new boolean[N + 1];
		for (int i = 0; i < M; i++) {
			int a_i = sc.nextInt();
			int b_i = sc.nextInt();

			if (a_i == 1) {
				a[b_i] = true;
			}
			if (b_i == 1) {
				a[a_i] = true;
			}
			if (a_i == N) {
				b[b_i] = true;
			}
			if (b_i == N) {
				b[a_i] = true;
			}
		}
		boolean answer = false;
		for (int i = 1; i <= N; i++) {
			if (a[i] && b[i]) {
				answer = true;
				break;
			}
		}
		if (answer) {
			System.out.println("POSSIBLE");
		} else {
			System.out.println("IMPOSSIBLE");
		}
	}
}