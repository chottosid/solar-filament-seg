# Publishing checklist (about ten minutes)

Nothing has been pushed or uploaded yet. These are the commands that make the repository public and
make the weights download work. Run them from this directory as the `chottosid` GitHub account
(`gh auth status` should say so). The weights archive is outside the repository, in
`../solar-filament-seg-weights/`.

## 1. Check the working tree

```bash
git status --short            # should print nothing
git log --oneline | head -3
```

## 2. Create the public repository and push

```bash
gh repo create chottosid/solar-filament-seg --public --source=. --remote=origin \
  --description "Solar filament instance segmentation (Kaggle 2026): YOLO11 voting, box-seeded U-Net refiners, count-matched selection"
git push -u origin main
```

If the local branch is called `master`, use `git push -u origin master` and tell GitHub with
`gh repo edit --default-branch master`, or rename first with `git branch -m main`.

## 3. Publish the weights as a release asset

The default `WEIGHTS_URL` (in `filament/weights.py`) expects tag `v1.0` and this exact file name.

```bash
gh release create v1.0 ../solar-filament-seg-weights/solar-filament-seg-weights.zip \
  --title "v1.0: submitted weights" \
  --notes "17 weight files (7 detectors, 5 refiners, 5 extent-robust 48-epoch refiners). Verified by data/weights.sha256; python -m filament.weights --dest weights downloads and checks them."
```

The archive is about 1.06 GB (GitHub's limit is 2 GB per asset). Its SHA-256 is
`45768998d4a8776eca8aa0c06e1b52426a4e9301723c9c2736c89057959ef3b2`.

## 4. Verify that a stranger's download works

```bash
rm -rf /tmp/verify && git clone https://github.com/chottosid/solar-filament-seg /tmp/verify
cd /tmp/verify && python -m filament.weights --dest weights
# expected: "17 weight files in weights match data/weights.sha256"
```

Also open the repository page in a private browser window: the code, the README and the release
asset should be visible without logging in. The competition asks for the repository to be public at
the close (15 November 2026, 06:00 UTC).

## 5. Put the link in the Google form

`https://github.com/chottosid/solar-filament-seg`, the report PDF, and the team name.

If you publish under a different account or repository name, change `WEIGHTS_URL` in
`filament/weights.py` (one line), commit and push before running step 4.
