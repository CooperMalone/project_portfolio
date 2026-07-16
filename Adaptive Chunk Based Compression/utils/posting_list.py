from typing import Dict, List


def text_to_posting_list(
    text: str, doc_id: int | str = 0, normalize: bool = True
) -> Dict[str, Dict[int | str, List[int]]]:
    posting_list: Dict[str, Dict[int | str, List[int]]] = {}

    tokens = text.split()

    for position, token in enumerate(tokens):
        term = token.lower() if normalize else token

        if term not in posting_list:
            posting_list[term] = {}

        if doc_id not in posting_list[term]:
            posting_list[term][doc_id] = []

        posting_list[term][doc_id].append(position)

    return posting_list


def merge_posting_lists(
    posting_lists: List[Dict[str, Dict[int | str, List[int]]]]
) -> Dict[str, Dict[int | str, List[int]]]:
    merged: Dict[str, Dict[int | str, List[int]]] = {}

    for posting_list in posting_lists:
        for term, doc_positions in posting_list.items():
            if term not in merged:
                merged[term] = {}
            
            for doc_id, positions in doc_positions.items():
                if doc_id not in merged[term]:
                    merged[term][doc_id] = []
                merged[term][doc_id].extend(positions)

    return merged


def documents_to_posting_list(
    documents: List[str], normalize: bool = True
) -> Dict[str, Dict[int, List[int]]]:
    posting_lists = [
        text_to_posting_list(doc, doc_id=idx, normalize=normalize)
        for idx, doc in enumerate(documents)
    ]
    return merge_posting_lists(posting_lists)


def get_term_gaps(
    posting_list: Dict[str, Dict[int | str, List[int]]], term: str
) -> Dict[int | str, List[int]]:
    if term not in posting_list:
        return {}

    gaps: Dict[int | str, List[int]] = {}

    for doc_id, positions in posting_list[term].items():
        if len(positions) > 1:
            doc_gaps = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
            gaps[doc_id] = doc_gaps

    return gaps
