package am1;

import java.util.HashMap;
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
		int[][] edges = new int[V+3][];
		for (int i = 1; i <= V+2; i++) {
			edges[i] = new int[V+3];
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
		Position[] spiral = new Position[Vemb];
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
			for (int i = Vemb - 1; i >= 0; i--) {
				spiral[i] = new Position(px, py, map[py][px]);
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
		}

		// calc
		int[] ax = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
		int[] ay = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
		// Key=emb_id, Value=g_id
		Map<Integer, Integer> vMap = new HashMap<>();
		{
			int max_v1 = 0;
			int max_v2 = 0;
			int max_v3 = 0;
			long max = 0L;
			for (int v1 = 1; v1 <= V; v1++) {
				for (int v2 = v1 + 1; v2 <= V; v2++) {
					for (int v3 = v2 + 1; v3 <= V; v3++) {
						long tmp = edges[v1][v2] + edges[v2][v3] + edges[v3][v1];
						if (max < tmp) {
							max = tmp;
							max_v1 = v1;
							max_v2 = v2;
							max_v3 = v3;
						}
					}
				}
			}
			vMap.put(spiral[0].id, max_v1);
			vMap.put(spiral[1].id, max_v2);
			vMap.put(spiral[2].id, max_v3);
		}
		for (int s_id = 3; s_id < V; s_id += 2) {
			int max_v1 = 0;
			int max_v2 = 0;
			long max = -1L;
			for (int v1 = 1; v1 <= V+1; v1++) {
				if (vMap.containsValue(v1)) continue;
				for (int v2 = 1; v2 <= V+1; v2++) {
					if (vMap.containsValue(v2)) continue;
					if (v1 == v2) continue;
					Map<Integer, Integer> tmpVmap = new HashMap<>(vMap);
					tmpVmap.put(spiral[s_id].id, v1);
					tmpVmap.put(spiral[s_id+1].id, v2);
					int[] vs = {v1, v2};
					long sum = 0L;
					for (int oid = 0; oid < 2; oid++) {
						for (int ad = 0; ad < 8; ad++) {
							int tmp_x = spiral[s_id + oid].x + ax[ad];
							int tmp_y = spiral[s_id + oid].y + ay[ad];
							int add_id = map[tmp_y][tmp_x];
							if (tmpVmap.containsKey(add_id)) {
								sum += edges[vs[oid]][tmpVmap.get(add_id)];
							}
						}
					}
					if (max < sum) {
						max = sum;
						max_v1 = v1;
						max_v2 = v2;
					}
				}
			}
			vMap.put(spiral[s_id].id, max_v1);
			vMap.put(spiral[s_id+1].id, max_v2);
		}

		// output
		for (Map.Entry<Integer, Integer> entry : vMap.entrySet()) {
			if (entry.getValue() <= V) {
				System.out.println(entry.getValue() + " " + entry.getKey());
			}
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
}