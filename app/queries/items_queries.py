from typing import List, Optional
from app.data.models import Item, ItemCreate

# Simple in-memory store for demonstration and initial development
_items: List[Item] = [
    Item(id=1, name="Item A", description="First item"),
    Item(id=2, name="Item B", description="Second item"),
]
_next_id = max(i.id for i in _items) + 1 if _items else 1


def get_all_items() -> List[Item]:
    return _items


def get_item_by_id(item_id: int) -> Optional[Item]:
    for it in _items:
        if it.id == item_id:
            return it
    return None


def create_item(item_in: ItemCreate) -> Item:
    global _next_id
    item = Item(id=_next_id, name=item_in.name, description=item_in.description)
    _items.append(item)
    _next_id += 1
    return item
