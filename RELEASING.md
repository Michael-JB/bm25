# Release process

*Instructions for maintainers only.*

1. Update the version in `Cargo.toml` (and ensure it's mirrored to Cargo.lock).
2. Update `CHANGELOG.md` with the new version and changes.
3. Commit and push these updates with message "build: Prepare vX.Y.Z release".
4. Create a GitHub release (easiest to just create the git tag via the GH UI).

