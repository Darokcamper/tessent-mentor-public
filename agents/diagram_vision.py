"""Diagram Vision — uses Gemini multimodal to analyze DFT diagrams, waveforms,
and flowcharts from lesson screenshots.

Caches descriptions to avoid re-calling the vision API for the same image.
"""
import json
import re
from pathlib import Path
from typing import List, Optional

from core.llm import llm

# Cache directory for diagram descriptions
CACHE_DIR = Path("knowledge") / "diagram_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Screenshot directory (where OCR'd images live)
# Actual structure: knowledge/Tessent Atpg Core Topics/{module}.{name}/{lesson}/Screenshot (N).png
SCREENSHOT_BASE = Path("knowledge") / "Tessent Atpg Core Topics"


def _cache_path(module_num: int, image_name: str) -> Path:
    """Return the cache JSON path for a given screenshot."""
    safe_name = re.sub(r'[^\w\-.]', '_', image_name)
    return CACHE_DIR / f"{module_num}_{safe_name}.json"


def describe_diagram(image_bytes: bytes, context: str = "", mime: str = "image/png") -> str:
    """Analyze a diagram image and return a detailed text description.

    Args:
        image_bytes: Raw image bytes
        context: Optional context about what this image is (lesson title, etc.)
        mime: Image MIME type (default image/png)

    Returns:
        Detailed description of the diagram
    """
    prompt = (
        "You are a senior DFT engineer analyzing a diagram from a Tessent training video. "
        "Describe this image in detail:\n"
        "1. What type of diagram is it? (scan chain, timing waveform, flowchart, "
        "   architecture block, state machine, etc.)\n"
        "2. Label ALL components, signals, data paths, and annotations visible.\n"
        "3. What concept or process does this diagram illustrate?\n"
        "4. If there are timing markers, signal names, or values, list them all.\n"
        "5. If there are tables or data values in the image, transcribe them.\n\n"
        f"Context: {context}" if context else ""
    )

    try:
        resp = llm.invoke_vision(prompt, image_bytes, image_mime=mime)
        return resp.content if hasattr(resp, 'content') else str(resp)
    except Exception as exc:
        return f"[Vision analysis failed: {exc}]"


def get_cached_description(module_num: int, image_name: str) -> Optional[str]:
    """Return cached description if it exists."""
    path = _cache_path(module_num, image_name)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            return data.get('description')
        except Exception:
            return None
    return None


def cache_description(module_num: int, image_name: str, description: str, source: str = ""):
    """Cache a diagram description to disk."""
    path = _cache_path(module_num, image_name)
    data = {
        "module_num": module_num,
        "image_name": image_name,
        "source": source,
        "description": description,
    }
    try:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception:
        pass


def find_lab_screenshots(lab_name: str, exercise: str = None) -> List[Path]:
    """Find all screenshots for a lab exercise.
    
    Args:
        lab_name: e.g., "Lab2"
        exercise: e.g., "EX1" — if None, returns all exercises' screenshots
    
    Returns:
        List of screenshot paths
    """
    labs_base = SCREENSHOT_BASE / "10.Labs"
    matches = []
    
    if not labs_base.exists():
        return matches
    
    lab_folder = None
    for folder in labs_base.iterdir():
        if folder.is_dir() and folder.name.lower() == lab_name.lower():
            lab_folder = folder
            break
    
    if not lab_folder:
        return matches
    
    if exercise:
        # Specific exercise
        ex_folder = None
        for sub in lab_folder.iterdir():
            if sub.is_dir() and sub.name.lower() == exercise.lower():
                ex_folder = sub
                break
        if ex_folder:
            for img in sorted(ex_folder.glob("*.png")) + sorted(ex_folder.glob("*.jpg")):
                matches.append(img)
    else:
        # All exercises
        for sub in sorted(lab_folder.iterdir()):
            if sub.is_dir():
                for img in sorted(sub.glob("*.png")) + sorted(sub.glob("*.jpg")):
                    matches.append(img)
    
    return matches


def find_screenshots_for_lesson(module_num: int, lesson_title: str) -> List[Path]:
    """Find all screenshots matching a lesson.

    Looks in knowledge/Tessent Atpg Core Topics/{module_num}.*/ for images whose folder
    matches the lesson title.
    """
    # Normalize lesson title for matching
    norm = lambda s: re.sub(r'[^\w]', '', s.lower())

    matches = []
    if not SCREENSHOT_BASE.exists():
        return matches

    # Search pattern: Tessent Atpg Core Topics/{module_num}.*{lesson}/
    for folder in sorted(SCREENSHOT_BASE.iterdir()):
        if not folder.is_dir():
            continue
        # Check if folder name starts with module number (e.g., "1.", "2.", etc.)
        if not folder.name.startswith(f"{module_num}."):
            continue
        # Look for lesson subfolder
        for subfolder in sorted(folder.iterdir()):
            if not subfolder.is_dir():
                continue
            folder_norm = norm(subfolder.name)
            lesson_norm = norm(lesson_title)
            # Check if folder name contains lesson title or vice versa
            if lesson_norm in folder_norm or folder_norm in lesson_norm:
                for img in sorted(subfolder.glob("*.png")) + sorted(subfolder.glob("*.jpg")):
                    matches.append(img)
    return matches


def analyze_lab_diagrams(lab_name: str, exercise: str = None) -> str:
    """Analyze all diagrams for a lab and return formatted descriptions.
    
    Args:
        lab_name: e.g., "Lab2"
        exercise: e.g., "EX1" — if None, analyzes all exercises
    
    Returns:
        Formatted string with all diagram descriptions, separated by exercise
    """
    labs_base = SCREENSHOT_BASE / "10.Labs"
    if not labs_base.exists():
        return ""
    
    lab_folder = None
    for folder in labs_base.iterdir():
        if folder.is_dir() and folder.name.lower() == lab_name.lower():
            lab_folder = folder
            break
    
    if not lab_folder:
        return ""
    
    parts = []
    
    if exercise:
        # Specific exercise
        ex_folder = None
        for sub in lab_folder.iterdir():
            if sub.is_dir() and sub.name.lower() == exercise.lower():
                ex_folder = sub
                break
        if ex_folder:
            parts.append(f"=== {ex_folder.name} ===")
            parts.extend(_analyze_images_in_folder(ex_folder, lab_name, exercise))
    else:
        # All exercises
        for sub in sorted(lab_folder.iterdir()):
            if sub.is_dir():
                parts.append(f"\n=== {sub.name} ===")
                parts.extend(_analyze_images_in_folder(sub, lab_name, sub.name))
    
    return "\n\n".join(parts)


def _analyze_images_in_folder(folder: Path, lab_name: str, exercise: str) -> List[str]:
    """Analyze all images in a folder, returning formatted descriptions."""
    results = []
    images = sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg"))
    
    for img_path in images:
        # Check cache first (use lab_name + exercise as cache key)
        cache_key = f"{lab_name}_{exercise}_{img_path.name}"
        cached = get_cached_description(0, cache_key)  # module_num=0 for labs
        if cached:
            results.append(f"[{img_path.name}]\n{cached}")
            continue
        
        # Analyze with vision
        try:
            image_bytes = img_path.read_bytes()
            description = describe_diagram(
                image_bytes,
                context=f"Lab: {lab_name}, Exercise: {exercise}",
            )
            # Cache the result
            cache_description(0, cache_key, description, str(img_path))
            results.append(f"[{img_path.name}]\n{description}")
        except Exception as exc:
            results.append(f"[{img_path.name}]\n[Failed to analyze: {exc}]")
    
    return results


def analyze_lesson_diagrams(module_num: int, lesson_title: str) -> str:
    """Analyze all diagrams for a lesson and return formatted descriptions.

    Uses cache to avoid re-analyzing previously processed images.
    """
    screenshots = find_screenshots_for_lesson(module_num, lesson_title)
    if not screenshots:
        return ""

    parts = []
    for img_path in screenshots:
        # Check cache first
        cached = get_cached_description(module_num, img_path.name)
        if cached:
            parts.append(f"[{img_path.name}]\n{cached}")
            continue

        # Analyze with vision
        try:
            image_bytes = img_path.read_bytes()
            description = describe_diagram(
                image_bytes,
                context=f"Module {module_num}, Lesson: {lesson_title}",
            )
            # Cache the result
            cache_description(module_num, img_path.name, description, str(img_path))
            parts.append(f"[{img_path.name}]\n{description}")
        except Exception as exc:
            parts.append(f"[{img_path.name}]\n[Failed to analyze: {exc}]")

    return "\n\n".join(parts)