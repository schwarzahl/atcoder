package arc085;

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class Main {
	public static void main(String[] args) {
		Main main = new Main();
		main.solveE();
	}
	public static int N;

	private void solveC() {
		Scanner sc = new Scanner(System.in);
		int N = sc.nextInt();
		int M = sc.nextInt();
		int time = N * 100 + M * 1800;
		int pi = 1;
		for (int i = 0; i < M; i++) {
			pi *= 2;
		}
		double prev = -1.0;
		double current = 0.0;
		int k = 1;
		while (Math.ceil(prev) != Math.ceil(current)) {
			prev = current;
			double tmp = 1.0 * k * time / pi;
			for (int d = 1; d < k; d++) {
				tmp *= (1.0 - 1.0 / pi);
			}
			current += tmp;
			k++;
		}
		System.out.println(100*Math.round(Math.ceil(current/100)));
	}

	private void solveD() {
		Scanner sc = new Scanner(System.in);
		int N = sc.nextInt();
		int Z = sc.nextInt();
		int W = sc.nextInt();
		int[] a = new int[N];
		for (int i = 0; i < N; i++) {
			a[i] = sc.nextInt();
		}
		int ans1 = Math.abs(a[N-1] - W);
		int ans2 = ans1;
		if (N > 1) {
			ans2 = Math.abs(a[N - 1] - a[N - 2]);
		}
		System.out.println(ans1 > ans2 ? ans1 : ans2);
	}

	private void solveE() {
		Scanner sc = new Scanner(System.in);
		N = sc.nextInt();
		int[] a = new int[N + 1];
		long sum = 0L;
		for (int i = 1; i <= N; i++) {
			a[i] = sc.nextInt();
			sum += a[i];
		}
		List<Node> list = new ArrayList<>();
		list.add(new Node(a));
		for (int x = 1; x <= N; x++) {
			List<Node> nextList = new ArrayList<>();
			for (Node node : list) {
				nextList.add(node);
				int[] tmp = new int[N + 1];
				for (int k = 1; k <= N; k++) {
					tmp[k] = node.arr[k];
					if (k % x == 0) {
						tmp[k] = 0;
					}
				}
				Node newNode = new Node(tmp);
				boolean contain = false;
				for (Node other : list) {
					if (newNode.equals(other)) {
						contain = true;
						break;
					}
				}
				if (contain) {
					nextList.add(newNode);
				}
			}
			list = nextList;
		}
		long max = 0L;
		for (Node node : list) {
			long tmp = 0L;
			for (int num : node.arr) {
				tmp += num;
			}
			if (max < tmp) {
				max = tmp;
			}
		}
		System.out.println(max);
	}
	private class Node {
		public int[] arr;
		public Node(int[] arr) {
			this.arr = new int[N + 1];
			for (int k = 1; k <= N; k++) {
				this.arr[k] = arr[k];
			}
		}
		public boolean equals(Node other) {
			for (int i = 1; i <= N; i++) {
				if (this.arr[i] != other.arr[i]) {
					return false;
				}
			}
			return true;
		}
	}
}