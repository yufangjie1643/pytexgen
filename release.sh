#!/usr/bin/env bash
set -euo pipefail

repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repository_dir"

action="${1:-check}"
case "$action" in
    build|check|upload) ;;
    *)
        echo "Usage: $0 [build|check|upload]" >&2
        exit 2
        ;;
esac

if [[ -x "$repository_dir/.venv/bin/python" ]]; then
    python_executable="$repository_dir/.venv/bin/python"
else
    python_executable="${PYTHON:-python3}"
fi

version="$(
    "$python_executable" -c \
        'import pathlib,re; text=pathlib.Path("pyproject.toml").read_text(); print(re.search(r"(?m)^version = \"([^\"]+)\"$", text).group(1))'
)"
release_dir="$repository_dir/dist/release-$version"

collect_artifacts() {
    shopt -s nullglob
    artifacts=(
        "$release_dir"/pytexgen-"$version".tar.gz
        "$release_dir"/pytexgen-"$version"-*.whl
    )
    shopt -u nullglob
    if (( ${#artifacts[@]} < 2 )); then
        echo "Expected an sdist and at least one wheel in $release_dir" >&2
        echo "Run: $0 build" >&2
        exit 1
    fi
}

check_artifacts() {
    collect_artifacts
    uvx twine check "${artifacts[@]}"
    (
        cd "$release_dir"
        sha256sum \
            pytexgen-"$version".tar.gz \
            pytexgen-"$version"-*.whl \
            > SHA256SUMS
    )
    echo "Release artifacts checked: $release_dir"
}

if [[ "$action" == "build" ]]; then
    mkdir -p "$release_dir"
    release_tmp="$repository_dir/build/release-tmp"
    mkdir -p "$release_tmp"
    TMPDIR="$release_tmp" \
        "$python_executable" -m build --no-isolation --outdir "$release_dir"
    check_artifacts
    exit 0
fi

check_artifacts
if [[ "$action" == "check" ]]; then
    exit 0
fi

if [[ -n "$(git status --porcelain)" ]]; then
    echo "Refusing to upload from a dirty Git worktree" >&2
    exit 1
fi

if [[ -f "$repository_dir/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$repository_dir/.env"
    set +a
fi
: "${TWINE_USERNAME:?TWINE_USERNAME is not configured}"
: "${TWINE_PASSWORD:?TWINE_PASSWORD is not configured}"

publish_artifacts=()
for artifact in "${artifacts[@]}"; do
    if [[ "$artifact" == *-linux_*.whl ]]; then
        echo "Skipping non-portable local Linux wheel: $(basename "$artifact")"
    else
        publish_artifacts+=("$artifact")
    fi
done
if (( ${#publish_artifacts[@]} == 0 )); then
    echo "No portable artifacts are available to upload" >&2
    exit 1
fi

echo "Uploading PyTexGen $version to PyPI"
uvx twine upload --non-interactive "${publish_artifacts[@]}"
