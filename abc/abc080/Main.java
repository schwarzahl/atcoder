package abc080;

import java.util.Scanner;

public class Main {
	public static void main(String[] args){
		Main main = new Main();
		main.solveC();
	}

	private void solveA() {
		Scanner sc = new Scanner(System.in);
		int N = sc.nextInt();
		int A = sc.nextInt();
		int B = sc.nextInt();
		System.out.println(N * A < B ? N * A : B);
	}

	private void solveB() {
		Scanner sc = new Scanner(System.in);
		int N = sc.nextInt();
		int tmp = N;
		int sum = 0;
		while (tmp > 0) {
			sum += tmp % 10;
			tmp /= 10;
		}
		if (N % sum == 0) {
			System.out.println("Yes");
		} else {
			System.out.println("No");
		}
	}

	private void solveC() {
		Scanner sc = new Scanner(System.in);
		int N = sc.nextInt();
		int[][] F = new int[N + 1][];
		for (int i = 1; i <= N; i++) {
			F[i] = new int[10];
			for (int j = 0; j < 10; j++) {
				F[i][j] = sc.nextInt();
			}
		}
		int[][] P = new int[N + 1][];
		for (int i = 1; i <= N; i++) {
			P[i] = new int[11];
			for (int j = 0; j <= 10; j++) {
				P[i][j] = sc.nextInt();
			}
		}
		long max = Long.MIN_VALUE;
		for (int pat = 1; pat < 1024; pat++) {
			long current = 0L;
			int[] M = new int[10];
			int tmp = pat;
			for (int i = 0; i < 10; i++) {
				M[i] = tmp % 2;
				tmp /= 2;
			}
			for (int i = 1; i <= N; i++) {
				int count = 0;
				for (int j = 0; j < 10; j++) {
					count += F[i][j] * M[j];
				}
				current += P[i][count];
			}
			if (max < current) {
				max = current;
			}
		}
		System.out.println(max);
	}

	private void solveD() {
		Scanner sc = new Scanner(System.in);
		int N = sc.nextInt();
		int C = sc.nextInt();
		int[][] M = new int[C + 1][];
		for (int j = 1; j <= C; j++) {
			M[j] = new int[100001];
		}
		for (int i = 1; i <= N; i++) {
			int s = sc.nextInt();
			int t = sc.nextInt();
			int c = sc.nextInt();
			for (int j = s; j <= t; j++) {
				M[c][j] = 1;
			}
		}
		int max = 0;
		for (int i = 1; i <= 100000; i++) {
			int sum = 0;
			for (int j = 1; j <= C; j++) {
				if (M[j][i] == 1) {
					sum++;
				}
			}
			if (max < sum) {
				max = sum;
			}
		}
		System.out.println(max);
	}
}