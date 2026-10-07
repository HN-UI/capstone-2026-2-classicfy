"use strict";
(() => {
  const study = window.LISTENING_STUDY;
  const el = id => document.getElementById(id);
  if (!study || study.schema_version !== 1) {
    el("global-message").textContent = "평가 자료가 준비되지 않았습니다. prepare_listening_study.py로 생성한 참가자 pack을 열어 주세요.";
    el("start").disabled = true; return;
  }
  const features = [["tempo", "전체 빠르기"], ["rubato", "국소적인 빠르기 변화"], ["dynamics", "강약의 크기와 대비"], ["articulation", "음의 연결·분리"], ["pedaling", "잔향·울림"]];
  const fields = ["overall", ...features.map(f => f[0]), "confidence"];
  const scale = [["1", "1 · 거의 없음"], ["2", "2 · 작음"], ["3", "3 · 분명함"], ["4", "4 · 큼"], ["5", "5 · 매우 큼"], ["unclear", "판단 어려움"]];
  const choices = [["X", "X"], ["Y", "Y"], ["tie", "둘이 비슷함"], ["unclear", "판단 어려움"]];
  let participant = "", position = 0, answers = {}, storageAvailable = true;
  const schedule = () => study.participants[participant];
  const task = () => schedule()[position];
  const storageKey = p => `classicfy-listening-${study.study_id}-${p}`;
  el("task-count").textContent = study.task_count;
  function pause() { document.querySelectorAll("audio").forEach(a => a.pause()); }
  function persist() {
    try { localStorage.setItem(storageKey(participant), JSON.stringify({ position, answers })); storageAvailable = true; }
    catch (e) { storageAvailable = false; }
    el("status").textContent = storageAvailable ? "이 브라우저에 저장됨" : "자동 저장 불가 · JSON 저장 버튼을 사용하세요";
  }
  function update(field, value) {
    answers[task().task_id] = { ...(answers[task().task_id] || {}), [field]: value, updated_at: new Date().toISOString() };
    persist();
  }
  function validateAnswers(value, p) {
    if (!value || typeof value !== "object" || Array.isArray(value)) throw Error("응답 형식이 올바르지 않습니다.");
    const tasks = study.participants[p];
    for (const [id, answer] of Object.entries(value)) {
      const t = tasks.find(t => t.task_id === id);
      if (!t || !answer || typeof answer !== "object" || Array.isArray(answer)) throw Error("알 수 없는 문항이 있습니다.");
      for (const field of fields) {
        if (answer[field] === undefined) continue;
        const allowed = field === "confidence" ? ["1", "2", "3"] : (t.mode === "cross" ? choices.map(x => x[0]) : scale.map(x => x[0]));
        if (!allowed.includes(answer[field])) throw Error("평가 척도가 올바르지 않습니다.");
      }
      if (answer.comment !== undefined && (typeof answer.comment !== "string" || answer.comment.length > 10000)) throw Error("메모 형식이 올바르지 않습니다.");
    }
  }
  function rating(name, title, options) {
    const wrap = document.createElement("fieldset"); wrap.className = "rating";
    const legend = document.createElement("legend"); legend.textContent = title; wrap.append(legend);
    const row = document.createElement("div"); row.className = "choices";
    for (const [value, label] of options) {
      const lab = document.createElement("label"); lab.className = "choice";
      const input = document.createElement("input"); input.type = "radio"; input.name = name; input.value = value;
      input.checked = (answers[task().task_id] || {})[name] === value;
      input.addEventListener("change", () => update(name, value));
      lab.append(input, document.createTextNode(label)); row.append(lab);
    }
    wrap.append(row); el("questions").append(wrap);
  }
  function render() {
    pause(); const t = task(); el("message").textContent = "";
    el("progress").textContent = `${participant} · ${position + 1} / ${schedule().length} · ${t.task_id}`;
    el("progress-bar").max = schedule().length; el("progress-bar").value = position;
    el("prompt").textContent = t.mode === "pair" ? "X와 Y의 연주 해석은 얼마나 다르게 들리나요?" : "A와 연주 경향이 더 비슷하게 들리는 후보는 어느 쪽인가요?";
    el("detail").textContent = t.mode === "pair" ? "같은 작품의 같은 악보 구간입니다. 구간 1의 X/Y, 구간 2의 X/Y를 비교한 뒤 두 구간을 종합해 평가하세요." : "X와 Y는 같은 작품의 같은 악보 구간이고, A는 다른 작품입니다. 곡의 선율보다 빠르기 변화·강약·연결감·울림의 경향을 비교하세요.";
    el("audio").replaceChildren();
    for (const [label, clips] of Object.entries(t.stimuli)) {
      const box = document.createElement("div"); box.className = "stimulus";
      const h = document.createElement("h3"); h.textContent = label; box.append(h);
      for (const clip of clips) {
        const p = document.createElement("p"); p.className = "clip-label"; p.textContent = `구간 ${clip.part}`;
        const audio = document.createElement("audio"); audio.controls = true; audio.preload = "metadata"; audio.src = clip.audio;
        audio.setAttribute("aria-label", `${label} 구간 ${clip.part}`);
        audio.addEventListener("play", () => document.querySelectorAll("audio").forEach(other => { if (other !== audio) other.pause(); }));
        audio.addEventListener("error", () => { el("message").textContent = `${label} 구간 ${clip.part}을 재생할 수 없습니다. pack의 audio 폴더를 확인하세요. 이 문항은 판단 어려움으로 기록할 수 있습니다.`; });
        box.append(p, audio);
      }
      el("audio").append(box);
    }
    el("questions").replaceChildren();
    if (t.mode === "pair") {
      rating("overall", "전체적으로 느껴지는 차이", scale);
      features.forEach(([name, label]) => rating(name, `${label}의 차이`, scale));
    } else {
      rating("overall", "A와 전반적인 연주 경향이 더 비슷한 후보", choices);
      features.forEach(([name, label]) => rating(name, `${label}에서 A와 더 비슷한 후보`, choices));
    }
    rating("confidence", "이 판단에 대한 확신", [["1", "낮음"], ["2", "보통"], ["3", "높음"]]);
    el("comment").value = (answers[t.task_id] || {}).comment || "";
    el("prev").disabled = position === 0; el("next").textContent = position === schedule().length - 1 ? "평가 마치기" : "다음";
    persist();
  }
  function payload() { return { schema_version: 1, study_id: study.study_id, participant, exported_at: new Date().toISOString(), schedule: schedule(), answers }; }
  function download(content, type, extension) {
    const url = URL.createObjectURL(new Blob([content], { type })); const a = document.createElement("a");
    a.href = url; a.download = `classicfy-listening-${study.study_id}-${participant}.${extension}`; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
  }
  function exportJson() { download(JSON.stringify(payload(), null, 2), "application/json", "json"); }
  function exportCsv() {
    const columns = ["study_id", "participant", "position", "task_id", "mode", ...fields, "comment"];
    const escape = x => '"' + String(x ?? "").replaceAll('"', '""') + '"';
    const rows = schedule().map((t, i) => { const a = answers[t.task_id] || {}; return [study.study_id, participant, i + 1, t.task_id, t.mode, ...fields.map(f => a[f]), a.comment]; });
    download("\uFEFF" + [columns, ...rows].map(r => r.map(escape).join(",")).join("\n") + "\n", "text/csv;charset=utf-8", "csv");
  }
  function home() { pause(); ["study", "done"].forEach(id => el(id).classList.add("hidden")); el("welcome").classList.remove("hidden"); }
  el("start").onclick = () => {
    participant = el("participant").value; el("global-message").textContent = "";
    if (!study.participants[participant]) { el("global-message").textContent = "참가자 번호를 선택하세요."; return; }
    try {
      const stored = JSON.parse(localStorage.getItem(storageKey(participant)) || "{}");
      answers = stored.answers || {}; validateAnswers(answers, participant);
      position = Number.isInteger(stored.position) && stored.position >= 0 ? Math.min(stored.position, schedule().length - 1) : 0;
    } catch (e) {
      if (e instanceof SyntaxError || e.message.startsWith("응답") || e.message.startsWith("알 수") || e.message.startsWith("평가") || e.message.startsWith("메모")) {
        el("global-message").textContent = "임시 응답을 읽을 수 없습니다. 이전 JSON이 있으면 불러와 주세요. 현재 응답은 덮어쓰지 않았습니다."; return;
      }
      answers = {}; position = 0; storageAvailable = false;
    }
    el("welcome").classList.add("hidden"); el("study").classList.remove("hidden"); render();
  };
  el("comment").oninput = () => update("comment", el("comment").value);
  el("ratings").onsubmit = e => e.preventDefault();
  el("prev").onclick = () => { position--; render(); };
  el("next").onclick = () => {
    const a = answers[task().task_id] || {};
    if (fields.some(f => !a[f])) { el("message").textContent = "모든 항목에 답해 주세요. 듣기 어렵다면 판단 어려움을 선택할 수 있어요."; return; }
    if (position + 1 < schedule().length) { position++; render(); }
    else {
      pause(); persist(); el("study").classList.add("hidden"); el("done").classList.remove("hidden");
      el("completion").textContent = `${participant} · ${schedule().length}문항 완료. JSON 파일을 진행자에게 전달하세요.`;
    }
    window.scrollTo(0, 0);
  };
  el("back").onclick = () => { el("done").classList.add("hidden"); el("study").classList.remove("hidden"); render(); };
  ["export-json", "finish-json"].forEach(id => el(id).onclick = exportJson);
  ["export-csv", "finish-csv"].forEach(id => el(id).onclick = exportCsv);
  ["exit", "home"].forEach(id => el(id).onclick = home);
  el("import").onchange = async e => {
    el("global-message").textContent = "";
    try {
      const file = e.target.files[0]; if (!file) return;
      if (file.size > 2 * 1024 * 1024) throw Error("응답 파일이 너무 큽니다.");
      const data = JSON.parse(await file.text()); const p = data.participant;
      if (data.schema_version !== 1 || data.study_id !== study.study_id || !study.participants[p]) throw Error("다른 평가의 응답 파일입니다.");
      if (JSON.stringify(data.schedule) !== JSON.stringify(study.participants[p])) throw Error("평가 순서가 일치하지 않습니다.");
      validateAnswers(data.answers, p);
      if (localStorage.getItem(storageKey(p)) && !window.confirm(`${p}의 이 브라우저 임시 응답을 불러온 파일로 교체할까요?`)) return;
      const tasks = study.participants[p]; const next = tasks.findIndex(t => fields.some(f => !(data.answers[t.task_id] || {})[f]));
      localStorage.setItem(storageKey(p), JSON.stringify({ position: next < 0 ? tasks.length - 1 : next, answers: data.answers }));
      el("participant").value = p; el("global-message").textContent = "응답을 불러왔습니다. 평가 시작 / 이어하기를 눌러 주세요.";
    } catch (error) { el("global-message").textContent = `불러오지 못했습니다: ${error.message}`; }
    e.target.value = "";
  };
})();
