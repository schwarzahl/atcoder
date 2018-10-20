package abc061;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;
public class Main {
	public static void main(String[] args){
		Main main = new Main();
		main.solveD();
	}

	private void solveA() {
		Scanner sc = new Scanner(System.in);
		int a = sc.nextInt();
		int b = sc.nextInt();
		int c = sc.nextInt();
		if (a <= c && c <= b) {
			System.out.println("Yes");
		} else {
			System.out.println("No");
		}
	}

	private void solveB() {
		Scanner sc = new Scanner(System.in);
		int N = sc.nextInt();
		int M = sc.nextInt();

		int[] num = new int[N + 1];
		for (int i = 0; i < M; i++) {
			num[sc.nextInt()]++;
			num[sc.nextInt()]++;
		}
		for (int j = 1; j <= N; j++) {
			System.out.println(num[j]);
		}
	}

	private void solveC() {
		Scanner sc = new Scanner(System.in);
		int N = sc.nextInt();
		long K = sc.nextLong();

		Map<Integer, Long> map = new HashMap<>();

		for (int i = 0; i < N; i++) {
			int key = sc.nextInt();
			long value = sc.nextLong();
			if (map.containsKey(key)) {
				value += map.get(key);
			}
			map.put(key, value);
		}
		long index = 0;
		for (Integer key : map.keySet().stream().sorted().toArray(Integer[]::new)) {
			index += map.get(key);
			if (K <= index) {
				System.out.println(key);
				break;
			}
		}
	}

	private void solveD() {
		Scanner sc = new Scanner(System.in);
		int N = sc.nextInt();
		int M = sc.nextInt();

		Set<Edge> edgeSet = new HashSet<Edge>();
		for (int i = 0; i < M; i++) {
			Edge e = new Edge();
			e.src = sc.nextInt();
			e.dest = sc.nextInt();
			e.cost = sc.nextLong();
			edgeSet.add(e);
		}

		Map<Integer, Long> pointMap = new HashMap<>();
		pointMap.put(1, 0L);
		Long ans = null;
		for (int calcNum = 1; calcNum < 2 * N; calcNum++) {
			for (Edge e : edgeSet) {
				if (pointMap.containsKey(e.src)) {
					long destValue = Long.MIN_VALUE;
					if (pointMap.containsKey(e.dest)) {
						destValue = pointMap.get(e.dest);
					}
					long nextValue = pointMap.get(e.src) + e.cost;
					if (destValue < nextValue) {
						pointMap.put(e.dest, nextValue);
					}
				}
			}
			if (calcNum == N - 1) {
				ans = pointMap.get(N);
			}
		}
		if (ans.equals(pointMap.get(N))) {
			System.out.println(ans);
		} else {
			System.out.println("inf");
		}
	}

	class Edge {
		public int src;
		public int dest;
		public long cost;
	}
}