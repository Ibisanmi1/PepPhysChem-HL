# Hugging Face Space — deploy and push

**Space:** [Ibisanmi1/PepPhysChem-HL](https://huggingface.co/spaces/Ibisanmi1/PepPhysChem-HL)  
**App entry:** `app.py` (declared in `README.md` YAML frontmatter). The Hub rejects pushes if **`short_description` is longer than 60 characters** (keep the Space card line short; put detail in the README body).

---

## 1. Install the Hugging Face CLI

```bash
curl -LsSf https://hf.co/cli/install.sh | bash
```

Ensure `hf` is on your `PATH` (often `~/.local/bin`).

---

## 2. Set your token (one-time per machine)

Create a token with **write** access to Spaces:  
[https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

**Option A — script (recommended)**

```bash
cd /path/to/PepPhysChem-HL
chmod +x scripts/hf_set_token.sh scripts/hf_push_space.sh
export HF_TOKEN=hf_your_token_here
./scripts/hf_set_token.sh
# Or: ./scripts/hf_set_token.sh hf_your_token_here
```

**Option B — interactive**

```bash
hf auth login
# paste token when prompted; add --add-to-git-credential for Git pushes to hf.co
```

**Option C — environment only (CI)**

```bash
export HF_TOKEN=hf_...
```

Many tools read `HF_TOKEN` automatically; Git still needs either `--add-to-git-credential` (see Option A) or a credential helper.

---

## 3. Clone the Space (optional, empty checkout)

When Git asks for a **password**, use the **token** (not your Hugging Face account password):

```bash
git clone https://huggingface.co/spaces/Ibisanmi1/PepPhysChem-HL
```

Or download files without Git:

```bash
hf download Ibisanmi1/PepPhysChem-HL --repo-type=space
```

---

## 4. Push this repository to the Space remote

From your **development clone** (this GitHub repo), add the Space as a second remote and push:

```bash
cd /path/to/PepPhysChem-HL
./scripts/hf_push_space.sh
```

That pushes your current branch to `main` on the Space. To push another branch:

```bash
./scripts/hf_push_space.sh my-feature-branch
```

If Git prompts for credentials after `hf auth login --add-to-git-credential`, use:

- **Username:** your Hugging Face username  
- **Password:** the same `hf_...` token  

### Push rejected: `fetch first` / non-fast-forward

The Space’s `main` often has an initial commit (e.g. README) that your laptop clone does not have, so the first `git push` is rejected.

**Option A — keep Space history and merge it in (safest)**

```bash
git fetch hf
git merge hf/main --allow-unrelated-histories -m "Merge Hugging Face Space main"
# fix conflicts if any, then commit
./scripts/hf_push_space.sh
```

**Option B — make your `main` replace Space `main` (drops remote-only commits)**

Only if you are sure nothing important exists only on the Space:

```bash
HF_PUSH_FORCE=1 ./scripts/hf_push_space.sh
```

The script also prints these hints if a normal push fails.

### HF rejects binary files (Xet / pre-receive hook)

If you see:

`Your push was rejected because it contains binary files` / `Please use … xet`

the Space **will not accept** normal Git commits that include large blobs (e.g. `.pt` checkpoints, big `.png` exports, `output/`, `images/`).

**What we do in this repo:** those paths are listed in **`.gitignore`**. Remove them from Git’s index (files stay on your disk), commit, then push again. **Do not** remove `checkpoints/.gitkeep` from the index—that file keeps an empty `checkpoints/` folder in Git for the Space.

```bash
git rm -r --cached output images data/dataanalysis 2>/dev/null || true
git ls-files 'checkpoints/*.pt' 'checkpoints/*.pth' 2>/dev/null | xargs git rm --cached -f 2>/dev/null || true
git add .gitignore checkpoints/.gitkeep
git commit -m "Stop tracking binaries for Hugging Face Space"
./scripts/hf_push_space.sh
```

**After a slim push, the Space still needs weights** so `app.py` can run:

1. The repo tracks **`checkpoints/.gitkeep`** so `checkpoints/` exists after a push. In the Space: **Files and versions → Add file** and upload your `.pt` files into `checkpoints/` (e.g.  
   `Half_Life_cnn_bilstm_embedding_physchem_run1.pt` — the default recommended preset), **or**
2. Host weights on the **Model Hub** and extend the app to `hf_hub_download` on startup (custom work), **or**
3. Use **Git LFS** / **Xet** as in [HF Xet docs](https://huggingface.co/docs/hub/xet) if you insist on Git-tracking binaries.

### Still rejected after `git rm --cached`?

Older commits still contain those blobs, so the **entire push** is rejected. Your local `main` can stay as-is; only the Space needs a clean history.

**Recommended — script (one slim commit to Space `main`, no old binary history):**

```bash
# clean working tree; .gitignore must exclude checkpoints, output, images, etc.
chmod +x scripts/hf_push_space_orphan.sh
./scripts/hf_push_space_orphan.sh
```

That creates a temporary orphan branch from your **current working tree**, force-pushes it to `hf/main`, then returns you to your branch.

**Alternative — rewrite local history:** [`git filter-repo`](https://github.com/newren/git-filter-repo) with `--path` / `--invert-paths` on `checkpoints/` and other dirs, then push (heavier; affects the branch you filter).

---

## 5. Space build notes

- **Port:** Spaces set `PORT`; `app.py` already uses it.
- **Errors in UI:** On Space, `show_error` defaults to **off** unless you set `GRADIO_SHOW_ERROR=true` in Space **Settings → Repository secrets → Variables** (or Variables UI).
- **Checkpoints:** Large `.pt` files may exceed Git LFS / Space limits if committed. Prefer **Hub model repo** + runtime `huggingface_hub` download, or Space **persistent storage**, per your hosting plan.
- **RDKit / build time:** First build can be slow; if the Space fails on RDKit, consider a **Docker Space** with conda-forge RDKit (advanced).

---

## 6. Sync GitHub → Space (alternative)

In the Space **Settings**, you can connect a **GitHub** repository so pushes to GitHub trigger rebuilds, instead of using `git push` to `huggingface.co`. Use one primary workflow to avoid conflicting histories.
