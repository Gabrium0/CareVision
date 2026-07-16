import assert from "node:assert/strict";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { explainNode, loadGraph, queryGraph, shortestPath, vocabulary } from "../scripts/graphify.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));
const graph = loadGraph(path.join(here, "fixture.graph.json"));
const directedGraph = loadGraph(path.join(here, "directed.graph.json"));

test("extracts normalized vocabulary", () => {
  const words = vocabulary(graph);
  assert(words.includes("auth"));
  assert(words.includes("service"));
  assert(words.includes("database"));
});

test("runs deterministic BFS and DFS traversals", () => {
  const bfs = queryGraph(graph, "auth", { budget: 2000 });
  const dfs = queryGraph(graph, "auth", { dfs: true, budget: 2000 });
  assert.match(bfs, /Traversal: BFS/);
  assert.match(dfs, /Traversal: DFS/);
  assert.match(bfs, /NODE Database/);
});

test("truncates output to the requested budget", () => {
  assert.match(queryGraph(graph, "auth", { budget: 20 }), /truncated at approximately 20 tokens/);
});

test("finds shortest paths and reports disconnected nodes", () => {
  assert.match(shortestPath(graph, "auth", "database"), /Shortest path \(2 hops\)/);
  assert.match(shortestPath(graph, "auth", "lonely"), /No path found/);
});

test("reports missing query and path nodes", () => {
  assert.match(queryGraph(graph, "unfindable"), /No matching nodes/);
  assert.match(shortestPath(graph, "unfindable", "database"), /Could not find nodes/);
});

test("decodes parallel multigraph edges in explanations", () => {
  const output = explainNode(graph, "auth");
  assert.match(output, /WRITES confidence=EXTRACTED key=writes/);
  assert.match(output, /READS confidence=INFERRED key=reads/);
});

test("preserves directed traversal and reports incoming edges", () => {
  assert.match(shortestPath(directedGraph, "api", "queue"), /Shortest path \(1 hops\)/);
  assert.match(shortestPath(directedGraph, "queue", "api"), /No path found/);
  assert.match(explainNode(directedGraph, "queue"), /IN ApiHandler --ENQUEUES confidence=EXTRACTED--> WorkQueue/);
});

test("prefers exact concept labels over longer prose matches", () => {
  assert.match(explainNode(directedGraph, "ApiHandler"), /^NODE ApiHandler /);
});

test("caps high-degree explanations", () => {
  assert.match(explainNode(graph, "auth", 20), /truncated at approximately 20 tokens/);
});
