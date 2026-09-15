"""Fix-up script for a database already seeded by seed_demo_data.py before it applied
KNOWN_METADATA -- corrects the three demo papers' title/authors/year, deletes the existing
(garbled-references) review draft, and regenerates a fresh one from the same syntheses.

Cheaper than re-running seed_demo_data.py from scratch: this does NOT re-extract claims or
regenerate syntheses (the expensive, billed steps) -- only the paper metadata and the review
draft (whose References section is what actually reads paper.title/paper.authors) are redone.

Usage:

    DATABASE_URL=<target database> OPENAI_API_KEY=<key> python scripts/fix_demo_paper_metadata.py

Safe to run more than once: metadata correction is idempotent, and each run deletes the prior
review draft before generating a new one rather than accumulating duplicates.
"""

from __future__ import annotations

from seed_demo_data import KNOWN_METADATA, REVIEW_TOPIC
from sqlalchemy import delete, select

from lit_review_assistant.db.models import Paper, ReviewDraft, ReviewSentence, ReviewSentenceClaim, Synthesis
from lit_review_assistant.db.session import session_scope
from lit_review_assistant.llm.review import generate_review_draft


def fix_paper_metadata() -> None:
    with session_scope() as session:
        papers = session.scalars(select(Paper)).all()
        fixed = 0
        for paper in papers:
            known = KNOWN_METADATA.get(paper.file_name)
            if known is None:
                continue
            paper.title = known["title"]
            paper.authors = known["authors"]
            paper.year = known["year"]
            fixed += 1
            print(f"Fixed {paper.paper_key}: {known['title']!r}")
        if fixed == 0:
            print("No matching demo papers found -- nothing to fix.")


def regenerate_review_draft() -> None:
    with session_scope() as session:
        old_draft_ids = [d.id for d in session.scalars(select(ReviewDraft)).all()]
        if old_draft_ids:
            sentence_query = select(ReviewSentence).where(ReviewSentence.review_draft_id.in_(old_draft_ids))
            sentence_ids = [s.id for s in session.scalars(sentence_query).all()]
            if sentence_ids:
                session.execute(
                    delete(ReviewSentenceClaim).where(ReviewSentenceClaim.review_sentence_id.in_(sentence_ids))
                )
            session.execute(delete(ReviewSentence).where(ReviewSentence.review_draft_id.in_(old_draft_ids)))
            session.execute(delete(ReviewDraft).where(ReviewDraft.id.in_(old_draft_ids)))
            print(f"Deleted {len(old_draft_ids)} old review draft(s).")

    with session_scope() as session:
        synthesis_ids = [s.id for s in session.scalars(select(Synthesis)).all()]
        if not synthesis_ids:
            print("No syntheses found -- can't generate a review draft.")
            return
        draft = generate_review_draft(session, topic=REVIEW_TOPIC, synthesis_ids=synthesis_ids)
        if draft is None:
            print("No review draft was created.")
        else:
            print(f"Generated review draft: {draft.title!r}")


def main() -> None:
    print("Fixing paper metadata...")
    fix_paper_metadata()

    print("\nRegenerating review draft...")
    regenerate_review_draft()


if __name__ == "__main__":
    main()
