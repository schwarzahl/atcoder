package agc014;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
public class Main {
	public static void main(String[] args){
		Main main = new Main();
		main.solveD();
	}

	private void solveA() {
		Scanner sc = new Scanner(System.in);
		long a = sc.nextLong();
		long b = sc.nextLong();
		long c = sc.nextLong();
		int n = 0;
		while (true) {
			if (a % 2 == 1 || b % 2 == 1 || c % 2 == 1) {
				System.out.println(n);
				return;
			}
			if (a == b && b == c) {
				System.out.println("-1");
				return;
				
			}
			n++;
			long nextA = (b + c) / 2;
			long nextB = (a + c) / 2;
			long nextC = (a + b) / 2;
			a = nextA;
			b = nextB;
			c = nextC;
		}
	}

	private void solveB() {
		Scanner sc = new Scanner(System.in);
		int N = sc.nextInt();
		int M = sc.nextInt();
		int[] count = new int[N];
		for (int i = 0; i < M; i++) {
			count[sc.nextInt() - 1]++;
			count[sc.nextInt() - 1]++;
		}
		int oddNum = 0;
		for (int num : count) {
			if (num % 2 == 1) {
				oddNum++;
			}
		}
		if (oddNum > 1) {
			System.out.println("NO");
		} else {
			System.out.println("YES");
		}
	}

	private void solveC() {
		Scanner sc = new Scanner(System.in);
		int H = sc.nextInt();
		int W = sc.nextInt();
		int K = sc.nextInt();
		sc.nextLine();
		int[][] dist_map = new int[H][];
		int start_r = -1;
		int start_c = -1;
		for (int r = 0; r < H; r++) {
			dist_map[r] = new int[W];
			String line = sc.nextLine();
			for (int c = 0; c < W; c++) {
				char panel = line.charAt(c);
				if (panel == 'S') {
					start_r = r;
					start_c = c;
					dist_map[r][c] = 1;
				} else if (panel == '#') {
					dist_map[r][c] = -1;
				} else {
					dist_map[r][c] = H * W * 2;
				}
			}
		}
		if (search(start_r, start_c, 0, dist_map, H, W, K) == 0) {
			System.out.println(1);
			return;
		}
		int min_dist = H + W;
		for (int r = 0; r < H; r++) {
			for (int c = 0; c < W; c++) {
				if (dist_map[r][c] <= K && dist_map[r][c] >= 0) {
					if (min_dist > r) {
						min_dist = r;
					}
					if (min_dist > c) {
						min_dist = c;
					}
					if (min_dist > H - 1 - r) {
						min_dist = H - 1 - r;
					}
					if (min_dist > W - 1 - c) {
						min_dist = W - 1 - c;
					}
				}
			}
		}
		System.out.println((min_dist - 1) / K + 2);
	}

	private int search(int r, int c, int dist, int[][] map, int limit_r, int limit_c, int K) {
		if (dist > K) {
			return -1;
		}
		if (r < 0 || c < 0 || r >= limit_r || c >= limit_c) {
			return 0;
		}
		if (dist < map[r][c]) {
			map[r][c] = dist;
			if (search(r - 1, c, dist + 1, map, limit_r, limit_c, K) == 0) {
				return 0;
			}
			if (search(r, c - 1, dist + 1, map, limit_r, limit_c, K) == 0) {
				return 0;
			}
			if (search(r + 1, c, dist + 1, map, limit_r, limit_c, K) == 0) {
				return 0;
			}
			if (search(r, c + 1, dist + 1, map, limit_r, limit_c, K) == 0) {
				return 0;
			}
		}
		return -1;
	}

	private void solveD() {
		Scanner sc = new Scanner(System.in);
		int N = sc.nextInt();
		Map<Integer, List<Integer>> mapList = new HashMap<>();
		for (int i = 0; i < N - 1; i++) {
			int a = sc.nextInt();
			int b = sc.nextInt();
			if (!mapList.containsKey(a)) {
				mapList.put(a, new ArrayList<Integer>());
			}
			if (!mapList.containsKey(b)) {
				mapList.put(b, new ArrayList<Integer>());
			}
			mapList.get(a).add(b);
			mapList.get(b).add(a);
		}
		if (judge(N, mapList)) {
			System.out.println("First");
		} else {
			System.out.println("Second");
		}
	}

	private boolean judge(int N, Map<Integer, List<Integer>> graph) {
		if (N % 2 == 1) {
			return true;
		}
		for (int a = 1; a <= N; a++) {
			List<Integer> adjList = graph.get(a);
			int oddPathCount = 0;
			for (int adjNode : adjList) {
				int length = path(a, adjNode, graph);
				if (length % 2 == 1) {
					oddPathCount++;
				}
			}
			if (oddPathCount > 1) {
				return true;
			}
		}
		return false;
	}

	private int path(int origin, int next, Map<Integer, List<Integer>> graph) {
		List<Integer> targetList = graph.get(next);
		if (targetList.size() > 2) {
			return 0;
		}
		for (int target : targetList) {
			if (target != origin) {
				int result = path(next, target, graph);
				if (result == 0) {
					return 0;
				} else {
					return result + 1;
				}
			}
		}
		return 1;
	}
}