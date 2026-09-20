from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import combinations, product
from math import sqrt
from pathlib import Path
from typing import Iterable, Mapping, Protocol, Sequence

from gliner import GLiNER


@dataclass(frozen=True, order=True)
class Triple:
    head: str
    relation: str
    tail: str


@dataclass(frozen=True)
class Evidence:
    kind: str
    triples: tuple[Triple, ...]
    text: str
    score: float


class TextGenerationModel(Protocol):
    def generate(self, prompt: str) -> str: ...


class TextEmbeddingModel(Protocol):
    def encode(self, text: str) -> Sequence[float]: ...


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions must match")
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def load_prompt(path: str | Path, fields: set[str]) -> str:
    template = Path(path).read_text(encoding="utf-8")
    missing = {field for field in fields if "{" + field + "}" not in template}
    if missing:
        raise ValueError(f"missing prompt fields: {sorted(missing)}")
    return template


class KnowledgeGraph:
    def __init__(self, triples: Iterable[Triple]) -> None:
        self.triples = tuple(sorted(set(triples)))
        self.nodes = {
            node for triple in self.triples for node in (triple.head, triple.tail)
        }
        self.neighbors: dict[str, set[str]] = defaultdict(set)
        for triple in self.triples:
            self.neighbors[triple.head].add(triple.tail)
            self.neighbors[triple.tail].add(triple.head)

    def candidate_domain(
        self,
        anchors: Sequence[str],
        max_hops: int,
    ) -> tuple[set[str], tuple[Triple, ...]]:
        distances: dict[str, int] = {}
        queue = deque((anchor, 0) for anchor in anchors if anchor in self.nodes)
        while queue:
            node, distance = queue.popleft()
            if distance > max_hops:
                continue
            previous = distances.get(node)
            if previous is not None and previous <= distance:
                continue
            distances[node] = distance
            if distance < max_hops:
                queue.extend(
                    (neighbor, distance + 1)
                    for neighbor in self.neighbors[node]
                )
        covered_nodes = set(distances)
        candidate_triples = tuple(
            triple
            for triple in self.triples
            if triple.head in covered_nodes and triple.tail in covered_nodes
        )
        return covered_nodes, candidate_triples

    def find_entity_paths(
        self,
        source: str,
        target: str,
        candidate_nodes: set[str],
        candidate_triples: Sequence[Triple],
        max_hops: int,
    ) -> list[tuple[str, ...]]:
        candidate_neighbors: dict[str, set[str]] = defaultdict(set)
        for triple in candidate_triples:
            candidate_neighbors[triple.head].add(triple.tail)
            candidate_neighbors[triple.tail].add(triple.head)
        paths: list[tuple[str, ...]] = []

        def search(current: str, path: tuple[str, ...]) -> None:
            if len(path) - 1 > max_hops:
                return
            if current == target:
                paths.append(path)
                return
            for neighbor in sorted(candidate_neighbors[current]):
                if neighbor in candidate_nodes and neighbor not in path:
                    search(neighbor, path + (neighbor,))

        if source in candidate_nodes and target in candidate_nodes:
            search(source, (source,))
        return paths

    @staticmethod
    def restore_relation_chains(
        entity_path: Sequence[str],
        candidate_triples: Sequence[Triple],
    ) -> list[tuple[Triple, ...]]:
        edge_facts: dict[frozenset[str], list[Triple]] = defaultdict(list)
        for triple in candidate_triples:
            edge_facts[frozenset((triple.head, triple.tail))].append(triple)
        choices = [
            edge_facts[frozenset((left, right))]
            for left, right in zip(entity_path, entity_path[1:])
        ]
        if not choices or any(not choice for choice in choices):
            return []
        return [tuple(chain) for chain in product(*choices)]


class KLUSE:
    def __init__(
        self,
        *,
        graph_triples: Iterable[Triple],
        relation_templates: Mapping[str, str],
        text_generation_model: TextGenerationModel,
        text_embedding_model: TextEmbeddingModel,
        gliner_model_checkpoint: str | Path,
        gliner_entity_labels: Sequence[str],
        gliner_inference_threshold: float,
        gliner_inference_batch_size: int,
        gliner_device: str,
        gliner_flat_ner: bool,
        gliner_multi_label: bool,
        initial_prompt_path: str | Path,
        final_prompt_path: str | Path,
        alignment_threshold: float,
        semantic_threshold: float,
        neighborhood_hops: int,
        max_path_hops: int,
        evidence_budget: int,
    ) -> None:
        checkpoint = Path(gliner_model_checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        if not gliner_entity_labels:
            raise ValueError("gliner_entity_labels must not be empty")
        if not 0.0 <= gliner_inference_threshold <= 1.0:
            raise ValueError("gliner_inference_threshold must be in [0, 1]")
        if gliner_inference_batch_size <= 0:
            raise ValueError("gliner_inference_batch_size must be positive")
        if not gliner_device:
            raise ValueError("gliner_device must not be empty")
        if not 0.0 <= alignment_threshold <= 1.0:
            raise ValueError("alignment_threshold must be in [0, 1]")
        if not 0.0 <= semantic_threshold <= 1.0:
            raise ValueError("semantic_threshold must be in [0, 1]")
        if neighborhood_hops < 0:
            raise ValueError("neighborhood_hops must be non-negative")
        if max_path_hops < 1:
            raise ValueError("max_path_hops must be positive")
        if evidence_budget <= 0 or evidence_budget % 2:
            raise ValueError("evidence_budget must be a positive even integer")

        self.graph = KnowledgeGraph(graph_triples)
        self.relation_templates = relation_templates
        self.text_generation_model = text_generation_model
        self.text_embedding_model = text_embedding_model
        self.gliner_entity_labels = list(gliner_entity_labels)
        self.gliner_inference_threshold = gliner_inference_threshold
        self.gliner_inference_batch_size = gliner_inference_batch_size
        self.gliner_flat_ner = gliner_flat_ner
        self.gliner_multi_label = gliner_multi_label
        self.alignment_threshold = alignment_threshold
        self.semantic_threshold = semantic_threshold
        self.neighborhood_hops = neighborhood_hops
        self.max_path_hops = max_path_hops
        self.evidence_budget = evidence_budget
        self.initial_prompt = load_prompt(initial_prompt_path, {"question"})
        self.final_prompt = load_prompt(
            final_prompt_path,
            {"question", "evidence"},
        )
        self.gliner = GLiNER.from_pretrained(
            str(checkpoint),
            map_location=gliner_device,
        )
        self.gliner.eval()

    def generate_initial_answer(self, question: str) -> str:
        prompt = self.initial_prompt.format(question=question)
        return self.text_generation_model.generate(prompt)

    def extract_entities(
        self,
        question: str,
        initial_answer: str,
    ) -> list[str]:
        outputs = self.gliner.run(
            [question, initial_answer],
            self.gliner_entity_labels,
            flat_ner=self.gliner_flat_ner,
            threshold=self.gliner_inference_threshold,
            multi_label=self.gliner_multi_label,
            batch_size=self.gliner_inference_batch_size,
        )
        entities: list[str] = []
        for predictions in outputs:
            for prediction in predictions:
                text = prediction.get("text")
                if text and text not in entities:
                    entities.append(text)
        return entities

    def similarity(self, left: str, right: str) -> float:
        return cosine_similarity(
            self.text_embedding_model.encode(left),
            self.text_embedding_model.encode(right),
        )

    def align_entities(self, entities: Sequence[str]) -> list[str]:
        graph_nodes = sorted(self.graph.nodes)
        anchors: list[str] = []
        for entity in entities:
            candidates = [
                (self.similarity(entity, node), node)
                for node in graph_nodes
            ]
            if not candidates:
                continue
            score, node = max(candidates)
            if score >= self.alignment_threshold and node not in anchors:
                anchors.append(node)
        return anchors

    def textualize(self, triples: Sequence[Triple]) -> str:
        fragments = []
        for triple in triples:
            template = self.relation_templates.get(
                triple.relation,
                "{head} -- {relation} --> {tail}",
            )
            fragments.append(
                template.format(
                    head=triple.head,
                    relation=triple.relation,
                    tail=triple.tail,
                )
            )
        return ", ".join(fragments)

    def collect_evidence(self, anchors: Sequence[str]) -> list[Evidence]:
        candidate_nodes, candidate_triples = self.graph.candidate_domain(
            anchors,
            self.neighborhood_hops,
        )
        local_evidence = [
            Evidence("local", (triple,), self.textualize((triple,)), 0.0)
            for triple in candidate_triples
        ]
        path_evidence: dict[tuple[Triple, ...], Evidence] = {}
        for source, target in combinations(anchors, 2):
            paths = self.graph.find_entity_paths(
                source,
                target,
                candidate_nodes,
                candidate_triples,
                self.max_path_hops,
            )
            for path in paths:
                chains = self.graph.restore_relation_chains(
                    path,
                    candidate_triples,
                )
                for chain in chains:
                    path_evidence.setdefault(
                        chain,
                        Evidence("path", chain, self.textualize(chain), 0.0),
                    )
        return local_evidence + list(path_evidence.values())

    def semantic_gate(
        self,
        question: str,
        evidence: Sequence[Evidence],
    ) -> list[Evidence]:
        retained = []
        for unit in evidence:
            score = self.similarity(question, unit.text)
            if score >= self.semantic_threshold:
                retained.append(
                    Evidence(unit.kind, unit.triples, unit.text, score)
                )
        return sorted(retained, key=lambda unit: (-unit.score, unit.text))

    def select_evidence(self, evidence: Sequence[Evidence]) -> list[Evidence]:
        quota = self.evidence_budget // 2
        local_evidence = [
            unit for unit in evidence if unit.kind == "local"
        ][:quota]
        path_evidence = [
            unit for unit in evidence if unit.kind == "path"
        ][:quota]
        return local_evidence + path_evidence

    def generate_final_answer(
        self,
        question: str,
        evidence: Sequence[Evidence],
    ) -> str:
        evidence_text = "; ".join(unit.text for unit in evidence)
        prompt = self.final_prompt.format(
            question=question,
            evidence=evidence_text,
        )
        return self.text_generation_model.generate(prompt)

    def answer(self, question: str) -> dict[str, object]:
        initial_answer = self.generate_initial_answer(question)
        entities = self.extract_entities(question, initial_answer)
        anchors = self.align_entities(entities)
        candidates = self.collect_evidence(anchors)
        gated_evidence = self.semantic_gate(question, candidates)
        selected_evidence = self.select_evidence(gated_evidence)
        final_answer = self.generate_final_answer(question, selected_evidence)
        return {
            "question": question,
            "initial_answer": initial_answer,
            "entities": entities,
            "anchors": anchors,
            "evidence": selected_evidence,
            "answer": final_answer,
        }
