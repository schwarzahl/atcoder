package agc012;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Scanner;
import java.util.Set;
public class Main {
	public static void main(String[] args){
		Main main = new Main();
		main.solveB();
	}

	private void solveA() {
		Scanner sc = new Scanner(System.in);
		int N = sc.nextInt();
		List<Long> list = new ArrayList<Long>();
		for (int i = 0; i < N * 3; i++) {
			list.add(sc.nextLong());
		}
		Collections.sort(list);
		long sum = 0;
		for (int i = N; i < N * 3; i++) {
			if (i % 2 == 0) {
				sum += list.get(i);
			}
		}
		System.out.println(sum);
	}

	private void solveB() {
		Scanner sc = new Scanner(System.in);
		int N = sc.nextInt();
		int M = sc.nextInt();
		Graph graph = new Graph(N);
		for (int i = 0; i < M; i++) {
			int a = sc.nextInt() - 1;
			int b = sc.nextInt() - 1;
			graph.join(a, b);
		}
		graph.calc();
		int Q = sc.nextInt();
		for (int i = 0; i < Q; i++) {
			int v = sc.nextInt() - 1;
			int d = sc.nextInt();
			int c = sc.nextInt();
			graph.putColor(v, d, c);
			//graph.outputString();
		}
		graph.outputString();
	}

	class Graph {
		Set<Long> set;
		int N;
		Map<Long, Integer> distMap;
		Map<Integer, Integer> colorMap;
		public Graph(int N) {
			set = new HashSet<Long>();
			this.N = N;
			colorMap = new HashMap<Integer, Integer>();
		}
		public void join(int a, int b) {
			long small, large;
			if (a < b) {
				small = a;
				large = b;
			} else {
				small = b;
				large = a;
			}
			set.add(small * N + large);
		}
		public boolean isJoin(int a, int b) {
			long small, large;
			if (a < b) {
				small = a;
				large = b;
			} else {
				small = b;
				large = a;
			}
			return set.contains(small * N + large);
		}
		public void putDist(int a, int b, int dist) {
			long small, large;
			if (a < b) {
				small = a;
				large = b;
			} else {
				small = b;
				large = a;
			}
			distMap.put(small * N + large, dist);
		}
		public int getDist(int a, int b) {
			long small, large;
			if (a < b) {
				small = a;
				large = b;
			} else {
				small = b;
				large = a;
			}
			if (!distMap.containsKey(small * N + large)) {
				return N + 1;
			}
			return distMap.get(small * N + large);
		}
		public void calc() {
			distMap = new HashMap<Long, Integer>();
			for (int i = 0; i < N; i++) {
				startCalcDist(i, i, 0);
			}
		}
		private void startCalcDist(int here, int origin, int d) {
			if (this.getDist(here, origin) > d) {
				this.putDist(here, origin, d);
			}
			for (int target = 0; target < N; target++) {
				if (here != target && this.isJoin(here, target)) {
					startCalcDist(target, origin, d + 1);
				}
			}
		}
		public void putColor(int v, int dist, int color) {
			for (int i = 0; i < N; i++) {
				if (getDist(v, i) <= dist) {
					colorMap.put(i, color);
				}
			}
		}
		public void outputString() {
			for (int i = 0; i < N; i++) {
				if (colorMap.containsKey(i)) {
					System.out.println(colorMap.get(i));
				} else {
					System.out.println(0);
				}
			}
			for (Entry<Long, Integer> e : distMap.entrySet()) {
				System.err.println(((e.getKey() / N) + 1) + "," + ((e.getKey() % N) + 1) + ":" + e.getValue());
			}
		}
	}
}