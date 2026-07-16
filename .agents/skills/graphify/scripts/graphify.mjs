#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const DEFAULT_BUDGET = 2000;

function endpointId(endpoint) {
  if (endpoint && typeof endpoint === "object") return String(endpoint.id ?? endpoint.label ?? "");
  return String(endpoint ?? "");
}

function splitWords(value) {
  const chunks = String(value ?? "").match(/[\p{L}\p{N}]+/gu) ?? [];
  const words = [];
  for (const chunk of chunks) {
    const parts = chunk.match(/[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+/g) ?? [chunk];
    for (const part of parts) {
      const word = part.toLocaleLowerCase();
      if (word.length >= 3 && word.length <= 30) words.push(word);
    }
  }
  return words;
}

export function loadGraph(graphPath) {
  const raw = JSON.parse(fs.readFileSync(graphPath, "utf8"));
  const nodes = new Map();
  for (const node of raw.nodes ?? []) nodes.set(String(node.id), { ...node, id: String(node.id) });

  const links = [];
  const adjacency = new Map([...nodes.keys()].map((id) => [id, []]));
  const incoming = new Map([...nodes.keys()].map((id) => [id, []]));
  for (const [index, link] of (raw.links ?? raw.edges ?? []).entries()) {
    const source = endpointId(link.source);
    const target = endpointId(link.target);
    if (!nodes.has(source) || !nodes.has(target)) continue;
    const edge = { ...link, source, target, index };
    links.push(edge);
    adjacency.get(source).push({ node: target, edge });
    incoming.get(target).push({ node: source, edge });
    if (!raw.directed) {
      adjacency.get(target).push({ node: source, edge });
      incoming.get(source).push({ node: target, edge });
    }
  }

  return {
    directed: Boolean(raw.directed),
    multigraph: Boolean(raw.multigraph),
    nodes,
    links,
    adjacency,
    incoming,
  };
}

export function vocabulary(graph) {
  const words = new Set();
  for (const node of graph.nodes.values()) {
    for (const word of splitWords(node.label)) words.add(word);
  }
  return [...words].sort((a, b) => a.localeCompare(b));
}

function queryTerms(value) {
  return [...new Set(splitWords(value))];
}

function nodeScores(graph, terms) {
  if (!terms.length) return [];
  const normalizedQuery = terms.join("");
  const frequency = new Map(terms.map((term) => [term, 0]));
  const labels = new Map();
  for (const [id, node] of graph.nodes) {
    const label = String(node.label ?? "").toLocaleLowerCase();
    labels.set(id, label);
    for (const term of terms) if (label.includes(term)) frequency.set(term, frequency.get(term) + 1);
  }
  const total = Math.max(graph.nodes.size, 1);
  const scored = [];
  for (const [id, label] of labels) {
    let score = 0;
    for (const term of terms) {
      if (label.includes(term)) score += Math.log((total + 1) / (frequency.get(term) + 1)) + 1;
    }
    const normalizedLabel = label.replace(/[^\p{L}\p{N}]+/gu, "");
    if (normalizedLabel === normalizedQuery) score += 100;
    if (score > 0) scored.push({ id, score });
  }
  return scored.sort((a, b) =>
    b.score - a.score ||
    String(graph.nodes.get(a.id)?.label ?? "").localeCompare(String(graph.nodes.get(b.id)?.label ?? "")) ||
    a.id.localeCompare(b.id)
  );
}

function bestNode(graph, value) {
  return nodeScores(graph, queryTerms(value))[0]?.id ?? null;
}

function label(graph, id) {
  return String(graph.nodes.get(id)?.label ?? id);
}

function edgeText(graph, edge, from, to) {
  const relation = edge.relation ?? edge.type ?? "related";
  const confidence = edge.confidence ? ` confidence=${edge.confidence}` : "";
  const key = graph.multigraph && edge.key !== undefined ? ` key=${edge.key}` : "";
  const connector = graph.directed ? `--${relation}${confidence}${key}-->` : `--${relation}${confidence}${key}--`;
  return `${label(graph, from)} ${connector} ${label(graph, to)}`;
}

function nodeText(node) {
  const source = node.source_file ?? node.source ?? "";
  const location = node.source_location ?? "";
  const community = node.community !== undefined ? ` community=${node.community}` : "";
  return `NODE ${node.label ?? node.id} [id=${node.id} source_file=${source} source_location=${location}${community}]`;
}

function truncate(text, budget) {
  const charBudget = Math.max(80, budget * 4);
  if (text.length <= charBudget) return text;
  return `${text.slice(0, Math.max(0, charBudget - 55))}\n... (truncated at approximately ${budget} tokens)`;
}

export function queryGraph(graph, question, options = {}) {
  const terms = queryTerms(question);
  const scored = nodeScores(graph, terms);
  const starts = scored.slice(0, 3).map(({ id }) => id);
  if (!starts.length) return `No matching nodes found for query terms: ${terms.join(", ") || "(none)"}`;

  const mode = options.dfs ? "DFS" : "BFS";
  const maxDepth = options.dfs ? 6 : 3;
  const visited = new Set();
  const traversedEdges = [];

  if (options.dfs) {
    const stack = starts.slice().reverse().map((id) => ({ id, depth: 0 }));
    while (stack.length) {
      const { id, depth } = stack.pop();
      if (visited.has(id) || depth > maxDepth) continue;
      visited.add(id);
      const neighbors = [...(graph.adjacency.get(id) ?? [])].sort((a, b) => label(graph, a.node).localeCompare(label(graph, b.node)));
      for (const next of neighbors.slice().reverse()) {
        if (!visited.has(next.node)) {
          traversedEdges.push({ from: id, to: next.node, edge: next.edge });
          stack.push({ id: next.node, depth: depth + 1 });
        }
      }
    }
  } else {
    const queue = starts.map((id) => ({ id, depth: 0 }));
    const queued = new Set(starts);
    while (queue.length) {
      const { id, depth } = queue.shift();
      visited.add(id);
      if (depth >= maxDepth) continue;
      const neighbors = [...(graph.adjacency.get(id) ?? [])].sort((a, b) => label(graph, a.node).localeCompare(label(graph, b.node)));
      for (const next of neighbors) {
        traversedEdges.push({ from: id, to: next.node, edge: next.edge });
        if (!queued.has(next.node)) {
          queued.add(next.node);
          queue.push({ id: next.node, depth: depth + 1 });
        }
      }
    }
  }

  const scoreMap = new Map(scored.map((item) => [item.id, item.score]));
  const ranked = [...visited].sort((a, b) =>
    (scoreMap.get(b) ?? 0) - (scoreMap.get(a) ?? 0) || label(graph, a).localeCompare(label(graph, b))
  );
  const lines = [`Traversal: ${mode} depth=${maxDepth} | Start: ${starts.map((id) => label(graph, id)).join(", ")} | ${visited.size} nodes found`];
  for (const id of ranked) lines.push(nodeText(graph.nodes.get(id)));
  const seenEdges = new Set();
  for (const item of traversedEdges) {
    const signature = graph.directed ? `${item.edge.index}:${item.from}:${item.to}` : String(item.edge.index);
    if (seenEdges.has(signature)) continue;
    seenEdges.add(signature);
    lines.push(`EDGE ${edgeText(graph, item.edge, item.from, item.to)}`);
  }
  return truncate(lines.join("\n"), options.budget ?? DEFAULT_BUDGET);
}

export function shortestPath(graph, sourceTerm, targetTerm, budget = DEFAULT_BUDGET) {
  const source = bestNode(graph, sourceTerm);
  const target = bestNode(graph, targetTerm);
  if (!source || !target) return `Could not find nodes matching: ${JSON.stringify(sourceTerm)} or ${JSON.stringify(targetTerm)}`;
  const queue = [source];
  const previous = new Map([[source, null]]);
  const via = new Map();
  while (queue.length && !previous.has(target)) {
    const current = queue.shift();
    for (const next of graph.adjacency.get(current) ?? []) {
      if (previous.has(next.node)) continue;
      previous.set(next.node, current);
      via.set(next.node, next.edge);
      queue.push(next.node);
    }
  }
  if (!previous.has(target)) return `No path found between ${JSON.stringify(sourceTerm)} and ${JSON.stringify(targetTerm)}`;
  const ids = [];
  for (let current = target; current !== null; current = previous.get(current)) ids.push(current);
  ids.reverse();
  const lines = [`Shortest path (${ids.length - 1} hops):`];
  for (let index = 0; index < ids.length; index += 1) {
    lines.push(nodeText(graph.nodes.get(ids[index])));
    if (index < ids.length - 1) lines.push(`EDGE ${edgeText(graph, via.get(ids[index + 1]), ids[index], ids[index + 1])}`);
  }
  return truncate(lines.join("\n"), budget);
}

export function explainNode(graph, term, budget = DEFAULT_BUDGET) {
  const id = bestNode(graph, term);
  if (!id) return `No node matching ${JSON.stringify(term)}`;
  const node = graph.nodes.get(id);
  const outgoing = graph.adjacency.get(id) ?? [];
  const incoming = graph.directed ? graph.incoming.get(id) ?? [] : [];
  const lines = [nodeText(node), `degree=${outgoing.length + incoming.length}`, "CONNECTIONS:"];
  for (const next of outgoing) lines.push(`OUT ${edgeText(graph, next.edge, id, next.node)}`);
  for (const prev of incoming) lines.push(`IN ${edgeText(graph, prev.edge, prev.node, id)}`);
  return truncate(lines.join("\n"), budget);
}

export function findGraph(start = process.cwd()) {
  let current = path.resolve(start);
  while (true) {
    const candidate = path.join(current, "graphify-out", "graph.json");
    if (fs.existsSync(candidate)) return candidate;
    const parent = path.dirname(current);
    if (parent === current) return null;
    current = parent;
  }
}

function parseCli(argv) {
  const args = [...argv];
  const command = args.shift();
  let graphPath;
  let budget = DEFAULT_BUDGET;
  let dfs = false;
  const positional = [];
  while (args.length) {
    const arg = args.shift();
    if (arg === "--graph") graphPath = path.resolve(args.shift() ?? "");
    else if (arg === "--budget") budget = Number.parseInt(args.shift() ?? "", 10);
    else if (arg === "--dfs") dfs = true;
    else positional.push(arg);
  }
  if (!Number.isFinite(budget) || budget <= 0) throw new Error("--budget must be a positive integer");
  return { command, graphPath, budget, dfs, positional };
}

export function runCli(argv = process.argv.slice(2)) {
  const { command, graphPath: explicitGraph, budget, dfs, positional } = parseCli(argv);
  if (command === "hook") {
    if (findGraph()) console.log("Graphify knowledge graph detected. Use the repository $graphify skill before broad source searches; its query/path/explain helper works inside the Codex sandbox.");
    return 0;
  }
  const graphPath = explicitGraph ?? findGraph();
  if (!graphPath) throw new Error("No graphify-out/graph.json found from the current directory upward");
  const graph = loadGraph(graphPath);
  if (command === "vocab") console.log(vocabulary(graph).join("\n"));
  else if (command === "query") console.log(queryGraph(graph, positional.join(" "), { dfs, budget }));
  else if (command === "path") console.log(shortestPath(graph, positional[0] ?? "", positional[1] ?? "", budget));
  else if (command === "explain") console.log(explainNode(graph, positional.join(" "), budget));
  else throw new Error("Usage: graphify.mjs vocab | query <terms> [--dfs] [--budget N] | path <source> <target> | explain <concept>");
  return 0;
}

if (process.argv[1] && pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url) {
  try {
    process.exitCode = runCli();
  } catch (error) {
    console.error(`graphify helper: ${error.message}`);
    process.exitCode = 1;
  }
}
