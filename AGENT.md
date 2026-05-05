# Agent Notes

- Do not commit generated mesh data, benchmark output, virtual environments, or local shortcut files such as `pytexgen.lnk`.
- For repository uploads, use the GitHub CLI authenticated account on this machine. Check with `gh auth status` first.
- After committing the intended files, push the current work to the main branch with the authenticated GitHub account, for example `git push origin HEAD:main`.
- If Git authentication is not wired to `gh`, run `gh auth setup-git` before pushing.
