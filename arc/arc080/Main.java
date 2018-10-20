package arc080;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Scanner;

public class Main {
	public static void main(String[] args) {
		Main main = new Main();
		main.solveE();
	}

	class Pair {
		int left;
		int right;
		Pair prev;
		Pair next;
		boolean del;

		public Pair(int left, int right, Pair prev) {
			this.left = left;
			this.right = right;
			this.prev = prev;
			this.del = false;
		}

		public int max() {
			return left > right ? left : right;
		}

		public int min() {
			return left > right ? right : left;
		}
	}

	private void solveE() {
		Scanner sc = new Scanner(System.in);
		int N = sc.nextInt();

		PriorityQueue<Pair> queue = new PriorityQueue(N, new Comparator() {
			@Override
			public int compare(Object o1, Object o2) {
				Pair p1 = (Pair)o1;
				Pair p2 = (Pair)o2;
				if (p1.min() > p2.min()) {
					return -1;
				}
				if (p1.min() < p2.min()) {
					return 1;
				}
				if (p1.max() > p2.max()) {
					return -1;
				}
				return 1;
			}
		});
		int prev = sc.nextInt();
		Pair prev_p = null;
		for (int i = 1; i < N; i++) {
			int tmp = sc.nextInt();
			Pair tmp_p = new Pair(prev, tmp, prev_p);
			queue.add(tmp_p);
			if (prev_p != null) {
				prev_p.next = tmp_p;
			}
			prev = tmp;
			prev_p = tmp_p;
		}
		ArrayDeque<Integer> answer = new ArrayDeque<>();
		while (!queue.isEmpty()) {
			Pair p = queue.poll();
			if (p.del) {
				continue;
			}
			if (p.prev != null) {
				p.prev.del = true;
			}
			if (p.next != null) {
				p.next.del = true;
			}
			if (p.prev != null && p.next != null) {
				Pair new_p = new Pair(p.prev.left, p.next.right, p.prev.prev);
				new_p.next = p.next.next;
				queue.add(new_p);
			}
			answer.addFirst(p.right);
			answer.addFirst(p.left);
		}
		while (answer.size() > 1) {
			System.out.print(answer.poll() + " ");
		}
		System.out.println(answer.poll());
	}

	private void solveD() {
		Scanner sc = new Scanner(System.in);
		int H = sc.nextInt();
		int W = sc.nextInt();
		int N = sc.nextInt();

		int[][] map = new int[H][];
		for (int h = 0; h < H; h++) {
			map[h] = new int[W];
		}
		int vx = 1;
		int ch = 0;
		int cw = 0;
		for (int c = 1; c <= N; c++) {
			int a = sc.nextInt();
			for (int j = 0; j < a; j++) {
				map[ch][cw] = c;
				cw += vx;
				if (cw == W) {
					ch++;
					cw = W - 1;
					vx = -1;
				}
				if (cw == -1) {
					ch++;
					cw = 0;
					vx = 1;
				}
			}
		}
		for (int h = 0; h < H; h++) {
			System.out.print(map[h][0]);
			for (int w = 1; w < W; w++) {
				System.out.print(" " + map[h][w]);
			}
			System.out.println();
		}
	}

	private void solveC() {
		Scanner sc = new Scanner(System.in);
		int N = sc.nextInt();

		int four = 0;
		int two = 0;
		for (int i = 0; i < N; i++) {
			int tmp = sc.nextInt();
			if (tmp % 4 == 0) {
				four++;
			} else if (tmp % 2 == 0) {
				two++;
			}
		}
		int num = 0;
		if (two > 1) {
			num += two - 1;
		}
		num += four * 2;
		if (num < N - 1) {
			System.out.println("No");
		} else {
			System.out.println("Yes");
		}
	}
}