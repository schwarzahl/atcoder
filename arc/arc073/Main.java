package arc073;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;
public class Main {
	public static void main(String[] args){
		Main main = new Main();
		main.solveD();
	}

	private void solveC() {
		Scanner sc = new Scanner(System.in);
		int n = sc.nextInt();
		long t = sc.nextLong();
		long ans = t * n;
		long prev = -t;
		for (int i = 0; i < n; i++) {
			long tmp = sc.nextLong();
			if (tmp - prev < t) {
				ans -= t - tmp + prev;
			}
			prev = tmp;
		}
		System.out.println(ans);
	}

	private void solveD() {
		Scanner sc = new Scanner(System.in);
		int n = sc.nextInt();
		long w = sc.nextLong();
		List<Item> list = new ArrayList<Item>();
		for (int i = 0; i < n; i++) {
			list.add(new Item(sc.nextInt(), sc.nextInt()));
		}
		Collections.sort(list, (o1, o2) -> o2.weight == o1.weight ? o2.value - o1.value : o2.weight - o1.weight);
		List<Node> nodeList = new ArrayList<Node>();
		nodeList.add(new Node(0L, w));
		long firstWeight = list.get(0).weight;
		for (int i = 0; i < n; i++) {
			Item item = list.get(i);
			List<Node> nextNodeList = new ArrayList<Node>();
			for (Node node : nodeList) {
				if (item.weight <= node.restWeight) {
					nextNodeList.add(new Node(node.currentValue + item.value, node.restWeight - item.weight));
				}
				if (node.restWeight <= (n - i) * (firstWeight + 3)) {
					nextNodeList.add(node);
				}
			}
			nodeList = nextNodeList;
		}
		long bestValue = 0;
		for (Node node : nodeList) {
			if (bestValue < node.currentValue) {
				bestValue = node.currentValue;
			}
		}
		System.out.println(bestValue);
	}

	class Item {
		public int weight;
		public int value;
		public Item(int weight, int value) {
			this.weight = weight;
			this.value = value;
		}
	}

	class Node {
		public long currentValue;
		public long restWeight;
		public Node(long currentValue, long restWeight) {
			this.currentValue = currentValue;
			this.restWeight = restWeight;
		}
	}
}