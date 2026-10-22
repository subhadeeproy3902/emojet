"""Chat-lexicon matcher, shared by slang_data.py, annotate.py and the training notebook (their builders inline
this file) so all three agree on what "this message uses <word>" means.

Rules (round 2 matched raw substrings, so "haga" fired inside "bhagao" and half the slang data was junk):
- Whole words only. A word is a run of Latin letters, or of Bengali / Devanagari letters + vowel signs.
- Romanized Bengali/Hindi is spelled many ways, so both sides are folded: lowercase, chch->ch, doubled letters
  collapsed (pacche/pachhe/pache, hagaa), sh->s, bh->v, ph->f, z->j, w->o, u->o, final y->i.
  English slang is only lowercased + collapsed (lmaoooo). Swear words are matched exactly: folding would turn
  Hindi "chhod" (leave it) into a swear word.
- Suffixes only from a list of real endings (Bengali -te/-er/-e/-ta, Hindi -on/-e/-en, Banglish brishtite),
  only for entries that allow them (inflect, the default) and only on stems of 5+ Latin letters or 3+ script
  characters. Any-ending matching made শীত (winter) fire on শীতল (cool).
- An entry only fires for its own language (English slang fires everywhere). A message tagged English that
  clearly reads as Banglish/Hinglish (marker words) counts as that language.
"""
import re

TOKEN = re.compile(r"[a-z]+|[ঀ-৿‌‍]+|[ऀ-ॿ‌‍]+")
FOLD = (("sh", "s"), ("bh", "v"), ("ph", "f"), ("z", "j"), ("w", "o"), ("u", "o"))
SUFFIXES = {"e", "te", "r", "er", "ta", "ti", "ke", "ra", "o", "i", "gulo", "on", "en", "a",            # Latin (folded)
            "ে", "র", "ের", "তে", "টা", "টি", "কে", "ও", "ই", "গুলো", "য়", "েই", "েও", "টাই",        # Bengali
            "ों", "ें", "े", "ी", "ा", "ो"}                                                          # Devanagari
BN_MARK = set("ami tumi tui amar tomar tor amra tomra korchi korbo korchis korcho achi achis acho ache hobe hoye hoyeche "
              "bhalo valo kotha jabo jacchi jachi keno kemon ekhon kichu chilo gechi geche pacche lagche khub onek sathe "
              "bolchi bolchis dekhchi korish hocche hoche niye theke".split())
HI_MARK = set("hai hain nahi nhi kya kyu kyun mera meri tera teri tum aap hum raha rahi rahe gaya gayi karo kuch bahut "
              "bohot abhi yaar mujhe tujhe hoga thi tha".split())
# a few real messages tagged Bengali are Spanish ("que se haga justicia"): two of these words = not Banglish/Hinglish
FOREIGN = set("que los las una del pero muy esta como cuando donde porque tambien nada todo usted ustedes".split())


def collapse(w):
    return re.sub(r"(.)\1+", r"\1", w)


def fold(w):
    w = collapse(w.replace("chch", "ch"))
    for a, b in FOLD:
        w = w.replace(a, b)
    return w[:-1] + "i" if w.endswith("y") else w


def _key(tok, mode):
    if tok[0] > "z":                     # Bengali / Devanagari: exact
        return tok
    return tok if mode == "exact" else (collapse(tok) if mode == "collapse" else fold(tok))


MODES = ("exact", "collapse", "fold")


class Lexicon:
    def __init__(self, entries):
        self.entries = entries
        self.index = {m: {} for m in MODES}           # mode -> first key -> [(entry, keys, term, suffix_ok)]
        for k, e in enumerate(entries):
            mode = "exact" if e["kind"] == "swear" else ("collapse" if e["lang"] == "en" or e.get("fold") is False else "fold")
            for term in list(e.get("roman", [])) + list(e.get("native", [])):
                toks = TOKEN.findall(term.lower())
                if not toks:
                    continue
                keys = tuple(_key(t, mode) for t in toks)
                last = keys[-1]
                suffix_ok = e.get("inflect", True) and len(last) >= (3 if last[0] > "z" else 5)
                self.index[mode].setdefault(keys[0], []).append((k, keys, term, suffix_ok))

    @staticmethod
    def lang_of(text, lang=None):
        """bn / hi / en for gating: script first, then the tag, then marker words for romanized text."""
        if re.search(r"[ঀ-৿]", text):
            return "bn"
        if re.search(r"[ऀ-ॿ]", text):
            return "hi"
        words = set(TOKEN.findall(text.lower()))
        if len(words & FOREIGN) >= 2:
            return "xx"
        if lang in ("bn", "hi"):
            return lang
        b, h = len(words & BN_MARK), len(words & HI_MARK)
        return "bn" if b > h else ("hi" if h > b else (lang or "en"))

    def find(self, text, lang=None):
        """[(entry index, matched term)], longest match per entry, in message order."""
        toks = TOKEN.findall(text.lower())
        if not toks:
            return []
        lang = self.lang_of(text, lang)
        seq = {m: [_key(t, m) for t in toks] for m in MODES}
        best = {}
        for m in MODES:
            ks = seq[m]
            for i, tok in enumerate(ks):
                cands = list(self.index[m].get(tok, ()))
                # a one-word stem with a suffix: brishtite -> brishti
                for cut in range(1, 5):
                    if len(tok) - cut >= 3 and tok[-cut:] in SUFFIXES:
                        cands += [c for c in self.index[m].get(tok[:-cut], ()) if c[3] and len(c[1]) == 1]
                for k, keys, term, suffix_ok in cands:
                    n = len(keys)
                    if i + n > len(ks):
                        continue
                    ok = all(ks[i + j] == keys[j] for j in range(n - 1))
                    lastt = ks[i + n - 1]
                    ok = ok and (lastt == keys[-1] or (suffix_ok and lastt.startswith(keys[-1]) and lastt[len(keys[-1]):] in SUFFIXES))
                    e = self.entries[k]
                    if ok and (e["lang"] == "en" or e["lang"] == lang):
                        if k not in best or len(term) > len(best[k][1]):
                            best[k] = (i, term)
        return [(k, t) for k, (i, t) in sorted(best.items(), key=lambda kv: kv[1][0])]

    def emojis(self, text, lang=None, short_words=4):
        """Meaning emojis for a message: topic entries always, reaction entries only in short messages."""
        short = len(text.split()) <= short_words
        out = []
        for k, _ in self.find(text, lang):
            e = self.entries[k]
            if e["kind"] == "topic" or (e["kind"] == "reaction" and short):
                out.append(e["emojis"])
        return out

    def notes(self, text, lang=None, limit=3):
        """'haga = poop / need to poop (crude, casual)' for the labeller's prompt."""
        return [f"{t} = {self.entries[k]['gloss']}" for k, t in self.find(text, lang)
                if self.entries[k]["kind"] != "address"][:limit]
