package am1;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

public class Main {
	public static void main(String[] args) {
		Main main = new Main();
		main.solve();
	}

	private void solve() {
		// input
		Scanner sc = new Scanner(System.in);
		int V = sc.nextInt();
		int E = sc.nextInt();
		int[][] edges = new int[V+1][];
		for (int i = 1; i <= V; i++) {
			edges[i] = new int[V+1];
		}
		for (int i = 0; i < E; i++) {
			int u = sc.nextInt();
			int v = sc.nextInt();
			int w = sc.nextInt();
			edges[u][v] = w;
			edges[v][u] = w;
		}
		int Vemb = sc.nextInt();
		int Eemb = sc.nextInt();
		for (int i = 0; i < Eemb; i++) {
			int a = sc.nextInt();
			int b = sc.nextInt();
		}

		// create embmap
		int N = (int)Math.round(Math.sqrt(Vemb));
		int[][] map = new int[N+2][N+2];
		for (int i = 0; i < N; i++) {
			map[i] = new int[N+2];
		}
		for (int i = 0; i < Vemb; i++) {
			map[(i/N)+1][0] = 0;
			map[(i/N)+1][(i%N)+1] = i+1;
			map[(i/N)+1][N+1] = 0;
		}
		for (int i = 0; i < N+2; i++) {
			map[0][i] = 0;
			map[N+1][i] = 0;
		}

		// run and create list
		List<Position> list = new ArrayList<>();
		{
			int px = 1;
			int py = 1;
			int pd = 0;
			int[] sx = {1, 0, -1, 0};
			int[] sy = {0, 1, 0, -1};
			int lx = 1;
			int rx = N;
			int uy = 1;
			int dy = N;
			while (list.size() < Vemb) {
				list.add(new Position(px, py, map[py][px]));
				px += sx[pd];
				py += sy[pd];
				if (sx[pd] > 0 && px == rx) {
					uy++;
					pd = (pd + 1) % 4;
				} else if (sy[pd] > 0 && py == dy) {
					rx--;
					pd = (pd + 1) % 4;
				} else if (sx[pd] < 0 && px == lx) {
					dy--;
					pd = (pd + 1) % 4;
				} else if (sy[pd] < 0 && py == uy) {
					lx++;
					pd = (pd + 1) % 4;
				}
			}
			Collections.reverse(list);
		}

		// calc
		int[] ax = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
		int[] ay = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
		// Key=emb_id, Value=g_id
		Map<Integer, Integer> vMap = new HashMap<>();
		for (Position pos : list) {
			long max = -1;
			int max_vid = 0;
			for (int vid = 1; vid <= V; vid++) {
				if (!vMap.containsValue(vid)) {
					long sum = 0L;
					for (int ad = 0; ad < 8; ad++) {
						int tmp_x = pos.x + ax[ad];
						int tmp_y = pos.y + ay[ad];
						int add_id = map[tmp_y][tmp_x];
						if (vMap.containsKey(add_id)) {
							sum += edges[vid][vMap.get(add_id)];
						}
					}
					if (max < sum) {
						max = sum;
						max_vid = vid;
					}
				}
			}
			vMap.put(pos.id, max_vid);
			if (vMap.size() >= V) {
				break;
			}
		}

		// output
		for (Map.Entry<Integer, Integer> entry : vMap.entrySet()) {
			System.out.println(entry.getValue() + " " + entry.getKey());
		}
	}

	class Position {
		public int x;
		public int y;
		public int id;
		public Position(int x, int y, int id) {
			this.x = x;
			this.y = y;
			this.id = id;
		}
	}

	class Edge {
		public int u;
		public int v;
		public int w;
		public Edge(int u, int v, int w) {
			this.u = u;
			this.v = v;
			this.w = w;
		}

	}
}