// FORMA — фитнес-тренер через веб-камеру.
// Поза: MediaPipe Pose Landmarker (в браузере). Подсчёт повторов, проверка техники
// и жесты управления — собственная логика на правилах (углы, пропорции, фазы движения).

import { PoseLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

/* =====================================================================
   1. Настройки
   ===================================================================== */
const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_URL = {
    lite: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    full: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
};
const IS_MOBILE = matchMedia("(pointer: coarse)").matches;
const MODEL_VARIANT = new URLSearchParams(location.search).get("model") || (IS_MOBILE ? "lite" : "full");

const DWELL_MS = 1200;        // сколько держать ладонь на кнопке
const FINISH_HOLD_MS = 2000;  // сколько держать скрещённые руки над головой
const SHOW_DELAY = 250;       // ошибка должна держаться столько мс, чтобы её показать
const HIDE_DELAY = 700;       // и пропасть на столько мс, чтобы её скрыть

const COLORS = { accent: "#3d63ff", good: "#2fd48a", warn: "#ffb020", bad: "#ff4d5e" };

/* =====================================================================
   2. Геометрия
   ===================================================================== */
const J = {
    NOSE: 0, L_EAR: 7, R_EAR: 8,
    L_SHOULDER: 11, R_SHOULDER: 12, L_ELBOW: 13, R_ELBOW: 14,
    L_WRIST: 15, R_WRIST: 16, L_INDEX: 19, R_INDEX: 20,
    L_HIP: 23, R_HIP: 24, L_KNEE: 25, R_KNEE: 26,
    L_ANKLE: 27, R_ANKLE: 28, L_HEEL: 29, R_HEEL: 30, L_FOOT: 31, R_FOOT: 32,
};

const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, v: Math.min(a.v, b.v) });

// Угол ABC в градусах, вершина в B
function angle(a, b, c) {
    const v1x = a.x - b.x, v1y = a.y - b.y, v2x = c.x - b.x, v2y = c.y - b.y;
    const d = Math.hypot(v1x, v1y) * Math.hypot(v2x, v2y);
    if (!d) return 180;
    return (Math.acos(clamp((v1x * v2x + v1y * v2y) / d, -1, 1)) * 180) / Math.PI;
}
// 0° — отрезок вертикален
const inclineFromVertical = (top, bottom) =>
    (Math.atan2(Math.abs(top.x - bottom.x), Math.abs(bottom.y - top.y)) * 180) / Math.PI;
// 0° — отрезок горизонтален
const inclineFromHorizontal = (a, b) =>
    (Math.atan2(Math.abs(a.y - b.y), Math.abs(a.x - b.x)) * 180) / Math.PI;

// Поза: raw — нормализованные точки (для отрисовки), pts — с поправкой на пропорции кадра (для углов)
class Body {
    constructor(raw, aspect) {
        this.raw = raw;
        this.pts = raw.map((l) => ({ x: l.x * aspect, y: l.y, v: l.visibility ?? 1 }));
    }
    p(i) { return this.pts[i]; }
    side(s) {
        const o = s === "L" ? 0 : 1;
        const idx = {
            ear: 7 + o, shoulder: 11 + o, elbow: 13 + o, wrist: 15 + o, index: 19 + o,
            hip: 23 + o, knee: 25 + o, ankle: 27 + o, heel: 29 + o, foot: 31 + o,
        };
        const out = { idx };
        for (const k in idx) out[k] = this.pts[idx[k]];
        return out;
    }
    bestSide() {
        const score = (o) => [11, 23, 25, 27].reduce((s, i) => s + this.pts[i + o].v, 0);
        return score(0) >= score(1) ? "L" : "R";
    }
    shoulderWidth() { return dist(this.pts[11], this.pts[12]); }
    torsoLen() { return dist(mid(this.pts[11], this.pts[12]), mid(this.pts[23], this.pts[24])) || 1e-3; }
    inside(i, th = 0.5) {
        const r = this.raw[i];
        return r.x > -0.02 && r.x < 1.02 && r.y > -0.02 && r.y < 1.02 && (r.visibility ?? 1) >= th;
    }
}

// Ракурс с гистерезисом: 'front' — лицом к камере, 'side' — боком
function trackView(prev, body) {
    const r = body.shoulderWidth() / body.torsoLen();
    if (r > 0.5) return "front";
    if (r < 0.32) return "side";
    return prev || "front";
}

// Сглаживание дрожания точек (экспоненциальное скользящее среднее)
class Smoother {
    constructor(alpha = 0.5) { this.alpha = alpha; this.prev = null; }
    apply(lms) {
        const a = this.alpha;
        if (!this.prev || this.prev.length !== lms.length) {
            this.prev = lms.map((l) => ({ x: l.x, y: l.y, z: l.z, visibility: l.visibility ?? 1 }));
            return this.prev;
        }
        this.prev = lms.map((l, i) => {
            const p = this.prev[i];
            return { x: p.x + a * (l.x - p.x), y: p.y + a * (l.y - p.y), z: p.z + a * (l.z - p.z), visibility: l.visibility ?? 1 };
        });
        return this.prev;
    }
    reset() { this.prev = null; }
}

/* =====================================================================
   3. Режим «ошибка»: каталог ошибок с конкретными подсказками
   kind: error — ошибка техники, warn — проблема с кадром, info — инструкция
   prio: чем меньше, тем важнее
   ===================================================================== */
const ISSUES = {
    not_visible: { kind: "warn", prio: 0, title: "Тебя не видно целиком", fix: "Отойди от камеры на 2–3 шага, чтобы в кадре были и голова, и стопы", say: "Отойди подальше, мне не видно тебя целиком" },
    face_camera: { kind: "warn", prio: 1, title: "Встань лицом к камере", fix: "Это упражнение я проверяю спереди — развернись к экрану", say: "Повернись лицом к камере" },
    turn_side: { kind: "warn", prio: 1, title: "Повернись боком к камере", fix: "Линию тела в планке видно только сбоку", say: "Повернись боком к камере" },

    squat_shallow: { title: "Неполный присед — не засчитано", fix: "Опускайся ниже: таз до уровня коленей, бедро параллельно полу", say: "Садись глубже, до параллели" },
    squat_valgus: { title: "Колени заваливаются внутрь", fix: "Разводи колени наружу — по линии носков", say: "Колени наружу" },
    squat_asym: { title: "Приседаешь с перекосом", fix: "Распредели вес поровну на обе ноги", say: "Держи вес на обеих ногах" },
    squat_toes: { title: "Колени выходят за носки", fix: "Отводи таз назад, как будто садишься на стул, вес на пятках", say: "Колени выходят за носки. Отводи таз назад" },
    squat_lean: { title: "Сильный наклон корпуса", fix: "Раскрой грудь, держи спину ровнее, смотри вперёд", say: "Спину ровнее, грудь вперёд" },
    squat_heels: { title: "Пятки отрываются от пола", fix: "Перенеси вес на пятки, стопы всей площадью в пол", say: "Пятки в пол" },

    jack_partial: { title: "Прыжок не засчитан", fix: "Одновременно подними руки над головой и расставь ноги шире плеч", say: "Руки вверх и ноги в стороны одновременно" },
    jack_arms_low: { title: "Руки не доходят до верха", fix: "Поднимай руки полностью над головой, почти до хлопка", say: "Руки выше головы" },
    jack_arm_lag: { title: "Одна рука отстаёт", fix: "Поднимай обе руки синхронно и одинаково высоко", say: "Руки синхронно" },
    jack_legs: { title: "Ноги расставлены слишком узко", fix: "Прыгай шире: стопы заметно шире плеч", say: "Ноги шире" },
    jack_bent: { title: "Руки согнуты в локтях", fix: "В верхней точке выпрямляй руки полностью", say: "Выпрями руки" },

    raise_low: { title: "Руки не дошли до уровня плеч", fix: "Поднимай руки до горизонтали — на высоту плеч", say: "Выше, до уровня плеч" },
    raise_high: { title: "Руки выше плеч", fix: "Останавливайся на уровне плеч: выше нагрузка уходит в трапецию", say: "Не выше плеч" },
    raise_asym: { title: "Руки на разной высоте", fix: "Поднимай обе руки синхронно до одной линии", say: "Руки на одной высоте" },
    raise_bent: { title: "Локти сильно согнуты", fix: "Держи руки почти прямыми, сгиб в локте совсем лёгкий", say: "Выпрями руки" },
    raise_shrug: { title: "Плечи тянутся к ушам", fix: "Опусти плечи вниз, шея длинная, работают только руки", say: "Опусти плечи" },
    raise_sway: { title: "Раскачиваешь корпус", fix: "Стой ровно, поднимай руки без рывка корпусом", say: "Не раскачивайся" },

    plank_get_down: { kind: "info", prio: 2, title: "Прими положение планки", fix: "Упор на предплечья или ладони, тело параллельно полу — таймер стартует сам", say: "Принимай упор лёжа" },
    plank_pike: { title: "Таз задран вверх", fix: "Опусти таз: плечи, таз и пятки на одной линии", say: "Опусти таз" },
    plank_sag: { title: "Таз проваливается вниз", fix: "Подкрути таз, напряги пресс и ягодицы", say: "Подними таз, напряги пресс" },
    plank_knees: { title: "Колени согнуты", fix: "Выпрями ноги и держи упор на носках", say: "Выпрями ноги" },
    plank_head: { title: "Голова опущена", fix: "Смотри в пол чуть впереди рук, шея продолжает линию спины", say: "Голову ровнее" },
};

function issue(code, extra = {}) {
    return { code, kind: "error", prio: 5, joints: [], ...ISSUES[code], ...extra };
}
const sideWord = (s) => (s === "L" ? "Левая" : "Правая");

/* =====================================================================
   4. Упражнения. update(body) → { issues, rep?, inPosition?, progress, metric, guides, debug }
   ===================================================================== */

// ---------- Приседания (лицом или боком) ----------
function createSquat() {
    let view = "front", phase = "up", minDepth = 1, thighRef = 0;
    return {
        update(b) {
            view = trackView(view, b);
            const s = b.bestSide(), S = b.side(s), Lf = b.side("L"), Rf = b.side("R");
            const side = view === "side";

            // Глубина: 1 — стоя, 0 — таз на уровне коленей (бедро параллельно полу)
            const thigh = side ? dist(S.hip, S.knee) : (dist(Lf.hip, Lf.knee) + dist(Rf.hip, Rf.knee)) / 2;
            thighRef = Math.max(thighRef * 0.999, thigh);
            const hipY = side ? S.hip.y : (Lf.hip.y + Rf.hip.y) / 2;
            const kneeY = side ? S.knee.y : (Lf.knee.y + Rf.knee.y) / 2;
            const depth = (kneeY - hipY) / (thighRef || 1e-3);

            const kl = angle(Lf.hip, Lf.knee, Lf.ankle), kr = angle(Rf.hip, Rf.knee, Rf.ankle);
            const issues = [];
            let rep = null;

            if (depth < 0.8) {
                if (!side) {
                    const kneeW = Math.abs(Lf.knee.x - Rf.knee.x), ankleW = Math.abs(Lf.ankle.x - Rf.ankle.x);
                    if (depth < 0.6 && kneeW < ankleW * 0.8)
                        issues.push(issue("squat_valgus", { joints: [J.L_KNEE, J.R_KNEE] }));
                    if (Math.abs(kl - kr) > 22) {
                        const lazy = kl > kr ? "L" : "R";
                        issues.push(issue("squat_asym", {
                            fix: `${lazy === "L" ? "Левое" : "Правое"} колено сгибается меньше — распредели вес поровну на обе ноги`,
                            joints: [lazy === "L" ? J.L_KNEE : J.R_KNEE],
                        }));
                    }
                } else {
                    const dir = Math.sign(S.foot.x - S.heel.x) || 1; // куда смотрят носки
                    const shin = dist(S.knee, S.ankle) || 1e-3;
                    if (depth < 0.5 && (S.knee.x - S.foot.x) * dir > 0.15 * shin)
                        issues.push(issue("squat_toes", { joints: [S.idx.knee, S.idx.foot] }));
                    if (inclineFromVertical(S.shoulder, S.hip) > 55)
                        issues.push(issue("squat_lean", { joints: [S.idx.shoulder, S.idx.hip] }));
                    if (S.foot.y - S.heel.y > 0.2 * shin)
                        issues.push(issue("squat_heels", { joints: [S.idx.heel] }));
                }
            }

            // Фазы: вверху → вниз → вверх = попытка
            if (phase === "up") {
                if (depth < 0.7) { phase = "down"; minDepth = depth; }
            } else {
                minDepth = Math.min(minDepth, depth);
                if (depth > 0.88) {
                    phase = "up";
                    rep = minDepth <= 0.3
                        ? { counted: true, errors: [] }
                        : { counted: false, errors: [issue("squat_shallow", { joints: [S.idx.hip] })] };
                }
            }

            // Направляющая: линия коленей, до которой надо опустить таз
            const guides = [];
            if (phase === "down") {
                const r = b.raw;
                const ky = side ? r[S.idx.knee].y : (r[25].y + r[26].y) / 2;
                const xs = side ? [r[S.idx.hip].x, r[S.idx.knee].x] : [r[23].x, r[24].x, r[25].x, r[26].x];
                guides.push({ type: "hline", y: ky, x1: Math.min(...xs) - 0.08, x2: Math.max(...xs) + 0.08, ok: depth <= 0.3, label: depth <= 0.3 ? "глубина ✓" : "таз до этой линии" });
            }

            return {
                issues, rep, guides,
                progress: clamp((1 - depth) / 0.7),
                metric: { label: "Глубина", value: Math.round(clamp((1 - depth) / 0.7, 0, 1.3) * 100) + "%" },
                debug: { view, depth: depth.toFixed(2), kneeL: Math.round(kl), kneeR: Math.round(kr), phase },
            };
        },
    };
}

// ---------- Джампинг-джек (лицом) ----------
function createJack() {
    let view = "front", phase = "closed", peak = null;
    return {
        update(b) {
            view = trackView(view, b);
            const Lf = b.side("L"), Rf = b.side("R");
            const tl = b.torsoLen(), sw = b.shoulderWidth() || 1e-3;
            const rl = (Lf.shoulder.y - Lf.wrist.y) / tl; // >0 — кисть выше плеча
            const rr = (Rf.shoulder.y - Rf.wrist.y) / tl;
            const raise = (rl + rr) / 2;
            const legs = dist(Lf.ankle, Rf.ankle) / sw;    // ширина стоп в ширинах плеч
            const issues = [];
            let rep = null;

            if (view === "side") {
                issues.push(issue("face_camera"));
                return { issues, progress: 0, metric: { label: "Амплитуда", value: "—" }, debug: { view } };
            }

            if (phase === "closed" && (raise > 0.3 || legs > 1.15)) {
                phase = "open";
                peak = { rl: -9, rr: -9, raise: -9, legs: 0, elbow: 180 };
            }
            if (phase === "open") {
                peak.rl = Math.max(peak.rl, rl);
                peak.rr = Math.max(peak.rr, rr);
                peak.legs = Math.max(peak.legs, legs);
                if (raise > peak.raise) {
                    peak.raise = raise;
                    peak.elbow = Math.min(angle(Lf.shoulder, Lf.elbow, Lf.wrist), angle(Rf.shoulder, Rf.elbow, Rf.wrist));
                }
                if (raise < -0.3 && legs < 0.95) {
                    phase = "closed";
                    const errors = [];
                    const counted = peak.raise >= 0.3 && peak.legs >= 1.0;
                    if (!counted) errors.push(issue("jack_partial"));
                    else {
                        if (Math.abs(peak.rl - peak.rr) > 0.35) {
                            const low = peak.rl < peak.rr ? "L" : "R";
                            errors.push(issue("jack_arm_lag", { title: `${sideWord(low)} рука отстаёт`, joints: [low === "L" ? J.L_WRIST : J.R_WRIST] }));
                        } else if (Math.min(peak.rl, peak.rr) < 0.7) {
                            errors.push(issue("jack_arms_low", { joints: [J.L_WRIST, J.R_WRIST] }));
                        }
                        if (peak.legs < 1.3) errors.push(issue("jack_legs", { joints: [J.L_ANKLE, J.R_ANKLE] }));
                        if (peak.elbow < 140) errors.push(issue("jack_bent", { joints: [J.L_ELBOW, J.R_ELBOW] }));
                    }
                    rep = { counted, errors };
                }
            }

            const progress = clamp((raise + 0.8) / 1.6) * 0.5 + clamp((legs - 0.7) / 0.8) * 0.5;
            return {
                issues, rep, progress, guides: [],
                metric: { label: "Амплитуда", value: Math.round(progress * 100) + "%" },
                debug: { view, raiseL: rl.toFixed(2), raiseR: rr.toFixed(2), legs: legs.toFixed(2), phase },
            };
        },
    };
}

// ---------- Подъём рук в стороны (лицом) ----------
function createRaise() {
    let view = "front", phase = "down", peakL = 0, peakR = 0, neckBase = 0;
    return {
        update(b) {
            view = trackView(view, b);
            const Lf = b.side("L"), Rf = b.side("R");
            const issues = [];
            let rep = null;

            if (view === "side") {
                issues.push(issue("face_camera"));
                return { issues, progress: 0, metric: { label: "Угол рук", value: "—" }, debug: { view } };
            }

            const aL = angle(Lf.hip, Lf.shoulder, Lf.wrist); // 0° — рука опущена, 90° — горизонталь
            const aR = angle(Rf.hip, Rf.shoulder, Rf.wrist);
            const avg = (aL + aR) / 2;
            const sw = b.shoulderWidth() || 1e-3;
            const neck = ((Lf.shoulder.y - Lf.ear.y) + (Rf.shoulder.y - Rf.ear.y)) / 2 / sw;
            if (phase === "down" && avg < 25) neckBase = neckBase ? neckBase + 0.05 * (neck - neckBase) : neck;

            if (avg > 45) {
                if (Math.max(aL, aR) > 115)
                    issues.push(issue("raise_high", { joints: [aL > aR ? J.L_WRIST : J.R_WRIST] }));
                if (Math.abs(aL - aR) > 20) {
                    const low = aL < aR ? "L" : "R";
                    issues.push(issue("raise_asym", { title: `${sideWord(low)} рука ниже`, joints: [low === "L" ? J.L_WRIST : J.R_WRIST] }));
                }
                const eL = angle(Lf.shoulder, Lf.elbow, Lf.wrist), eR = angle(Rf.shoulder, Rf.elbow, Rf.wrist);
                if (Math.min(eL, eR) < 130) issues.push(issue("raise_bent", { joints: [J.L_ELBOW, J.R_ELBOW] }));
                if (neckBase && neck < neckBase * 0.7) issues.push(issue("raise_shrug", { joints: [J.L_SHOULDER, J.R_SHOULDER] }));
                if (inclineFromVertical(mid(Lf.shoulder, Rf.shoulder), mid(Lf.hip, Rf.hip)) > 10)
                    issues.push(issue("raise_sway", { joints: [J.L_HIP, J.R_HIP] }));
            }

            if (phase === "down") {
                if (avg > 55) { phase = "up"; peakL = aL; peakR = aR; }
            } else {
                peakL = Math.max(peakL, aL);
                peakR = Math.max(peakR, aR);
                if (avg < 30) {
                    phase = "down";
                    rep = Math.min(peakL, peakR) >= 70
                        ? { counted: true, errors: [] }
                        : { counted: false, errors: [issue("raise_low", { joints: [J.L_WRIST, J.R_WRIST] })] };
                }
            }

            const guides = [];
            if (avg > 30) {
                const r = b.raw;
                const y = (r[11].y + r[12].y) / 2;
                const xs = [r[11].x, r[12].x, r[15].x, r[16].x];
                guides.push({ type: "hline", y, x1: Math.min(...xs) - 0.05, x2: Math.max(...xs) + 0.05, ok: avg >= 75 && avg <= 115, label: "уровень плеч" });
            }

            return {
                issues, rep, guides,
                progress: clamp(avg / 90),
                metric: { label: "Угол рук", value: Math.round(avg) + "°" },
                debug: { view, aL: Math.round(aL), aR: Math.round(aR), neck: neck.toFixed(2), neckBase: neckBase.toFixed(2), phase },
            };
        },
    };
}

// ---------- Планка (боком, на время) ----------
function createPlank() {
    let view = "side";
    return {
        update(b) {
            view = trackView(view, b);
            const S = b.side(b.bestSide());
            const tl = b.torsoLen();
            const issues = [];
            const tilt = inclineFromHorizontal(S.shoulder, S.ankle);
            const supported = S.elbow.y > S.shoulder.y - 0.02; // есть упор: локти ниже плеч
            const inPosition = view === "side" && tilt < 35 && supported;

            if (!inPosition) {
                issues.push(view === "front" && tilt < 35 ? issue("turn_side") : issue("plank_get_down"));
                return { issues, inPosition: false, progress: 0, guides: [], metric: { label: "Наклон", value: Math.round(tilt) + "°" }, debug: { view, tilt: Math.round(tilt) } };
            }

            // Насколько таз выше (+) или ниже (−) прямой «плечо — лодыжка»
            const dx = S.ankle.x - S.shoulder.x;
            const t = dx ? (S.hip.x - S.shoulder.x) / dx : 0.5;
            const lineY = S.shoulder.y + t * (S.ankle.y - S.shoulder.y);
            const off = (lineY - S.hip.y) / tl;

            if (off > 0.15) issues.push(issue("plank_pike", { joints: [S.idx.hip] }));
            else if (off < -0.12) issues.push(issue("plank_sag", { joints: [S.idx.hip] }));
            if (angle(S.hip, S.knee, S.ankle) < 150) issues.push(issue("plank_knees", { joints: [S.idx.knee] }));
            if ((S.ear.y - S.shoulder.y) / tl > 0.35) issues.push(issue("plank_head", { joints: [S.idx.ear] }));

            const r = b.raw;
            return {
                issues, inPosition: true,
                progress: clamp(1 - Math.abs(off) / 0.3),
                guides: [{ type: "seg", a: r[S.idx.shoulder], b: r[S.idx.ankle], ok: off <= 0.15 && off >= -0.12, label: "линия тела" }],
                metric: { label: "Линия тела", value: Math.round(angle(S.shoulder, S.hip, S.ankle)) + "°" },
                debug: { view, tilt: Math.round(tilt), off: off.toFixed(2) },
            };
        },
    };
}

const EXERCISES = [
    {
        id: "squat", name: "Приседания", icon: "🏋️", type: "reps", target: 10, view: "any", need: "full",
        desc: "Считаю повторы и проверяю глубину, колени и спину.",
        setup: "Встань лицом или боком к камере в 2–3 шагах, чтобы в кадр попало всё тело. Боком я вижу больше ошибок.",
        checks: ["Глубина: таз до уровня коленей", "Колени не заваливаются внутрь (лицом)", "Колени не уходят за носки (боком)", "Ровная спина, пятки на полу (боком)"],
        create: createSquat,
    },
    {
        id: "jack", name: "Джампинг-джек", icon: "⭐", type: "reps", target: 15, view: "front", need: "front",
        desc: "Прыжки «звёздочкой»: руки вверх, ноги в стороны.",
        setup: "Встань лицом к камере. Отойди так, чтобы было видно всё тело и немного места над головой.",
        checks: ["Руки полностью над головой", "Ноги шире плеч", "Руки поднимаются синхронно", "Прямые локти в верхней точке"],
        create: createJack,
    },
    {
        id: "raise", name: "Подъём рук в стороны", icon: "🙆", type: "reps", target: 12, view: "front", need: "upper",
        desc: "Разведение рук до уровня плеч — можно с бутылками воды.",
        setup: "Встань лицом к камере. Достаточно, чтобы в кадре были корпус и руки.",
        checks: ["Руки до уровня плеч, но не выше", "Обе руки на одной высоте", "Плечи не тянутся к ушам", "Корпус не раскачивается"],
        create: createRaise,
    },
    {
        id: "plank", name: "Планка", icon: "🧱", type: "hold", target: 30, view: "side", need: "full",
        desc: "Засчитываю только секунды с ровной линией тела.",
        setup: "Повернись боком к камере и расположись так, чтобы в кадре было тело целиком — и стоя, и в упоре.",
        checks: ["Плечи, таз и пятки на одной линии", "Таз не задран и не провисает", "Прямые ноги", "Шея продолжает линию спины"],
        create: createPlank,
    },
];

// Видно ли нужные для упражнения части тела
function inFrame(b, need) {
    const ok = (ids) => ids.every((i) => b.inside(i));
    if (need === "upper") return ok([11, 12, 13, 14, 15, 16, 23, 24]);
    if (need === "front") return ok([0, 11, 12, 23, 24, 25, 26, 27, 28]);
    const o = b.bestSide() === "L" ? 0 : 1;
    return ok([11 + o, 23 + o, 25 + o, 27 + o]);
}

/* =====================================================================
   5. Монитор ошибок: убирает мерцание, выбирает главную подсказку
   ===================================================================== */
class FormMonitor {
    constructor() { this.live = new Map(); this.flashItem = null; this.flashUntil = 0; }
    update(issues, now) {
        const newly = [];
        const seen = new Set();
        for (const is of issues) {
            seen.add(is.code);
            let st = this.live.get(is.code);
            if (!st) { st = { since: now, last: now, shown: false, issue: is }; this.live.set(is.code, st); }
            st.last = now;
            st.issue = is;
            if (!st.shown && now - st.since >= SHOW_DELAY) { st.shown = true; newly.push(is); }
        }
        for (const [code, st] of this.live) {
            if (seen.has(code)) continue;
            if ((!st.shown && now - st.last > 150) || now - st.last > HIDE_DELAY) this.live.delete(code);
        }
        const active = [...this.live.values()].filter((s) => s.shown).map((s) => s.issue).sort((a, b) => a.prio - b.prio);
        const flash = now < this.flashUntil ? this.flashItem : null;
        return { active, newly, current: active[0] || flash || null };
    }
    flash(is, ms, now) { this.flashItem = is; this.flashUntil = now + ms; }
}

/* =====================================================================
   6. Звук и голос
   ===================================================================== */
let audioCtx = null;
let muted = false;
function unlockAudio() {
    try {
        audioCtx ??= new (window.AudioContext || window.webkitAudioContext)();
        if (audioCtx.state === "suspended") audioCtx.resume();
    } catch { /* без звука */ }
}
function tone(freq, dur = 0.12, type = "sine", gain = 0.15, when = 0) {
    if (muted || !audioCtx) return;
    const t = audioCtx.currentTime + when;
    const o = audioCtx.createOscillator(), g = audioCtx.createGain();
    o.type = type;
    o.frequency.setValueAtTime(freq, t);
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(gain, t + 0.01);
    g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    o.connect(g).connect(audioCtx.destination);
    o.start(t);
    o.stop(t + dur + 0.02);
}
const sfx = {
    rep() { tone(660, 0.1, "triangle"); tone(990, 0.14, "triangle", 0.15, 0.08); },
    bad() { tone(220, 0.2, "sawtooth", 0.07); },
    tick() { tone(880, 0.08, "square", 0.06); },
    go() { tone(1320, 0.3, "triangle", 0.15); },
    select() { tone(520, 0.08, "sine", 0.12); tone(780, 0.1, "sine", 0.12, 0.07); },
    finish() { [523, 659, 784, 1047].forEach((f, i) => tone(f, 0.22, "triangle", 0.14, i * 0.12)); },
};

let lastSpeak = 0;
const lastSpeakByKey = new Map();
function say(text, key = text, force = false) {
    if (muted || !("speechSynthesis" in window) || !text) return;
    const now = performance.now();
    if (!force && (now - lastSpeak < 2500 || now - (lastSpeakByKey.get(key) ?? -1e9) < 7000)) return;
    lastSpeak = now;
    lastSpeakByKey.set(key, now);
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.lang = "ru-RU";
    u.rate = 1.1;
    const voice = speechSynthesis.getVoices().find((v) => v.lang.toLowerCase().startsWith("ru"));
    if (voice) u.voice = voice;
    speechSynthesis.speak(u);
}

/* =====================================================================
   7. История и рекорды (localStorage)
   ===================================================================== */
const STORE_KEY = "forma.history.v1";
function loadHistory() {
    try { return JSON.parse(localStorage.getItem(STORE_KEY)) || []; } catch { return []; }
}
function saveEntry(entry) {
    const h = loadHistory();
    h.push(entry);
    try { localStorage.setItem(STORE_KEY, JSON.stringify(h.slice(-300))); } catch { /* приватный режим */ }
}
const bestScore = (exId) => loadHistory().filter((e) => e.ex === exId).reduce((m, e) => Math.max(m, e.score), 0);

/* =====================================================================
   8. DOM и отрисовка
   ===================================================================== */
const $ = (s) => document.querySelector(s);
const stage = $("#stage");
const video = $("#camera");
const statusEl = $("#status");
const canvas = $("#overlay");
const ctx = canvas.getContext("2d");
const cursorEl = $("#cursor");
const cursorProg = cursorEl.querySelector(".prog");
const debugEl = $("#debug");

const BONES = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24], [23, 25], [25, 27], [24, 26], [26, 28], [27, 29], [29, 31], [27, 31], [28, 30], [30, 32], [28, 32]];
const JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32];

// Как видео вписано в экран — чтобы скелет и курсор совпадали с картинкой
function layout() {
    const sw = stage.clientWidth, sh = stage.clientHeight;
    const vw = video.videoWidth || 1280, vh = video.videoHeight || 720;
    const s = video.style.objectFit === "contain" ? Math.min(sw / vw, sh / vh) : Math.max(sw / vw, sh / vh);
    return { sw, sh, vw, vh, s, ox: (sw - vw * s) / 2, oy: (sh - vh * s) / 2 };
}
function toStage(lm, L) {
    let x = lm.x * L.vw * L.s + L.ox;
    if (app.mirror) x = L.sw - x;
    return { x, y: lm.y * L.vh * L.s + L.oy };
}
function updateFit() {
    if (!video.videoWidth) return;
    const va = video.videoWidth / video.videoHeight, sa = stage.clientWidth / stage.clientHeight;
    // Если пропорции сильно различаются (горизонтальная камера на вертикальном экране) — показываем кадр целиком
    video.style.objectFit = (va > 1.2 && sa < 0.9) || (va < 0.9 && sa > 1.2) ? "contain" : "cover";
}

function draw(now, body, bad, guides) {
    const dpr = window.devicePixelRatio || 1;
    const L = layout();
    const W = Math.round(L.sw * dpr), H = Math.round(L.sh * dpr);
    if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, L.sw, L.sh);
    if (!body || app.screen === "intro") return;

    const P = body.raw.map((p) => toStage(p, L));
    const vis = (i) => (body.raw[i].visibility ?? 1) > 0.4;
    const lw = Math.max(3, Math.min(L.sw, L.sh) / 140);
    const good = now < app.goodUntil;
    ctx.lineCap = "round";

    // Направляющие
    for (const g of guides) {
        const a = g.type === "hline" ? toStage({ x: g.x1, y: g.y }, L) : toStage(g.a, L);
        const c = g.type === "hline" ? toStage({ x: g.x2, y: g.y }, L) : toStage(g.b, L);
        ctx.save();
        ctx.setLineDash([lw * 3, lw * 2]);
        ctx.strokeStyle = g.ok ? COLORS.good : "rgba(255,255,255,.75)";
        ctx.lineWidth = lw * 0.8;
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(c.x, c.y); ctx.stroke();
        if (g.label) {
            ctx.setLineDash([]);
            ctx.font = `600 ${Math.round(lw * 4)}px Inter, sans-serif`;
            ctx.fillStyle = g.ok ? COLORS.good : "#fff";
            ctx.fillText(g.label, Math.min(a.x, c.x), Math.min(a.y, c.y) - lw * 2);
        }
        ctx.restore();
    }

    // Кости
    for (const [a, c] of BONES) {
        if (!vis(a) || !vis(c)) continue;
        const isBad = bad.has(a) || bad.has(c);
        ctx.strokeStyle = isBad ? COLORS.bad : good ? COLORS.good : "rgba(255,255,255,.85)";
        ctx.lineWidth = isBad ? lw * 1.4 : lw;
        ctx.beginPath(); ctx.moveTo(P[a].x, P[a].y); ctx.lineTo(P[c].x, P[c].y); ctx.stroke();
    }
    // Суставы
    for (const i of JOINTS) {
        if (!vis(i)) continue;
        const isBad = bad.has(i);
        if (isBad) {
            const pulse = lw * (3 + Math.sin(now / 110) * 1.2);
            ctx.fillStyle = "rgba(255,77,94,.28)";
            ctx.beginPath(); ctx.arc(P[i].x, P[i].y, pulse * 1.8, 0, Math.PI * 2); ctx.fill();
        }
        ctx.fillStyle = isBad ? COLORS.bad : good ? COLORS.good : COLORS.accent;
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(P[i].x, P[i].y, isBad ? lw * 1.8 : lw * 1.2, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    }
}

/* =====================================================================
   9. Жесты: курсор-ладонь и «скрещённые руки над головой»
   ===================================================================== */
function setRing(p) { cursorProg.style.strokeDashoffset = String(100.53 * (1 - p)); }

class HandCursor {
    constructor(el) { this.el = el; this.target = null; this.start = 0; this.blockUntil = 0; this.x = null; this.y = null; }
    block(ms) { this.blockUntil = performance.now() + ms; this.clearTarget(); }
    clearTarget() {
        if (this.target) { this.target.classList.remove("dwelling"); this.target.style.removeProperty("--dwell"); }
        this.target = null;
        setRing(0);
    }
    hide() { this.el.classList.remove("on"); this.clearTarget(); this.x = this.y = null; }
    update(b, now) {
        if (!b) return this.hide();
        // Рука, поднятая выше таза; если обе — та, что выше
        const hands = [[J.L_INDEX, J.L_WRIST, J.L_HIP], [J.R_INDEX, J.R_WRIST, J.R_HIP]]
            .filter(([ix, wr, hip]) => b.inside(ix) && b.inside(wr) && b.raw[ix].y < b.raw[hip].y)
            .map(([ix]) => b.raw[ix])
            .sort((a, c) => a.y - c.y);
        if (!hands.length) return this.hide();

        const p = toStage(hands[0], layout());
        const rect = stage.getBoundingClientRect();
        const tx = p.x + rect.left, ty = p.y + rect.top;
        this.x = this.x == null ? tx : this.x + 0.5 * (tx - this.x);
        this.y = this.y == null ? ty : this.y + 0.5 * (ty - this.y);
        this.el.style.transform = `translate(${this.x}px, ${this.y}px)`;
        this.el.classList.add("on");
        if (now < this.blockUntil) return;

        const hit = document.elementFromPoint(this.x, this.y)?.closest("[data-gesture]") || null;
        if (hit !== this.target) {
            this.clearTarget();
            this.target = hit;
            this.start = now;
            if (hit) hit.classList.add("dwelling");
            return;
        }
        if (!hit) return;
        const prog = clamp((now - this.start) / DWELL_MS);
        hit.style.setProperty("--dwell", prog);
        setRing(prog);
        if (prog >= 1) {
            this.clearTarget();
            this.block(1000);
            sfx.select();
            hit.click();
        }
    }
}

function armsCrossedOverhead(b) {
    if (!b.inside(J.L_WRIST) || !b.inside(J.R_WRIST)) return false;
    const nose = b.p(J.NOSE), lw = b.p(J.L_WRIST), rw = b.p(J.R_WRIST);
    return lw.y < nose.y && rw.y < nose.y && dist(lw, rw) < b.shoulderWidth() * 0.6;
}

/* =====================================================================
   10. Приложение: экраны и сценарий
   ===================================================================== */
const app = {
    screen: "intro",
    detector: null,
    smoother: new Smoother(0.5),
    body: null,
    lastSeen: 0,
    lastVideoTime: -1,
    mirror: true,
    def: null,
    ex: null,
    monitor: null,
    session: null,
    prepView: null,
    prepOkSince: 0,
    cd: null,
    goodUntil: 0,
    debug: false,
    fps: { frames: 0, since: 0, value: 0 },
    started: false,
};
const cursor = new HandCursor(cursorEl);

function show(name) {
    app.screen = name;
    document.body.dataset.screen = name;
    document.querySelectorAll("[data-screen]").forEach((s) => s.classList.toggle("active", s.dataset.screen === name && s.tagName === "SECTION"));
    cursor.block(900);
}

function setStatus(text) { statusEl.textContent = text; }

// ---------- Старт камеры и модели ----------
async function start(src) {
    unlockAudio();
    show("loading");
    $("#loading-text").textContent = src.file ? "Открываю видео…" : "Включаю камеру… Разреши доступ в браузере";
    try {
        if (src.file) {
            video.srcObject = null;
            video.src = URL.createObjectURL(src.file);
            video.loop = true;
            app.mirror = false;
        } else {
            if (!navigator.mediaDevices?.getUserMedia) throw Object.assign(new Error("no-media"), { name: "NoMedia" });
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
                audio: false,
            });
            video.srcObject = stream;
            app.mirror = true;
        }
        video.classList.toggle("mirror", app.mirror);
        await video.play();
        video.classList.add("live");
        updateFit();
        setStatus(src.file ? "Видео из файла" : "Камера подключена");

        if (!app.detector) {
            $("#loading-text").textContent = "Загружаю модель распознавания позы…";
            app.detector = await createDetector(MODEL_VARIANT);
        }
        if (!app.started) { app.started = true; requestAnimationFrame(loop); }
        renderMenu();
        show("menu");
        say("Выбери упражнение. Наведи ладонь на карточку", "menu", true);
    } catch (err) {
        console.error(err);
        const msg = {
            NotAllowedError: "Доступ к камере запрещён. Разреши его в адресной строке браузера (значок камеры) и попробуй снова.",
            NotFoundError: "Камера не найдена. Подключи камеру или загрузи видео с тренировкой.",
            NotReadableError: "Камера занята другим приложением (Zoom, Teams…). Закрой его и попробуй снова.",
            NoMedia: "Браузер не даёт доступ к камере. Открой сайт по https или через localhost.",
        }[err.name] || `Не удалось запустить: ${err.message || err}. Проверь интернет — модель загружается из сети.`;
        $("#error-text").textContent = msg;
        setStatus("Ошибка");
        show("error");
    }
}

async function createDetector(variant) {
    const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
    const make = (delegate) => PoseLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: MODEL_URL[variant] || MODEL_URL.full, delegate },
        runningMode: "VIDEO",
        numPoses: 1,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
    });
    try {
        return await make("GPU");
    } catch (e) {
        console.warn("GPU недоступен, работаю на CPU", e);
        return await make("CPU");
    }
}

// ---------- Главный цикл ----------
function loop() {
    requestAnimationFrame(loop);
    const now = performance.now();

    if (app.detector && video.readyState >= 2 && video.currentTime !== app.lastVideoTime) {
        app.lastVideoTime = video.currentTime;
        let res = null;
        try { res = app.detector.detectForVideo(video, now); } catch (e) { console.error(e); }
        const lms = res?.landmarks?.[0];
        if (lms) {
            app.body = new Body(app.smoother.apply(lms), (video.videoWidth || 16) / (video.videoHeight || 9));
            app.lastSeen = now;
        } else if (now - app.lastSeen > 300) {
            app.body = null;
            app.smoother.reset();
        }
        const f = app.fps;
        f.frames++;
        if (now - f.since > 1000) {
            f.value = Math.round((f.frames * 1000) / (now - f.since));
            f.frames = 0;
            f.since = now;
            setStatus(`${app.mirror ? "Камера" : "Видео"} · ${f.value} fps · ${app.body ? "человек в кадре" : "никого не вижу"}`);
        }
    }

    let bad = new Set(), guides = [];
    switch (app.screen) {
        case "menu":
        case "summary":
            cursor.update(app.body, now);
            break;
        case "prepare":
            cursor.update(app.body, now);
            tickPrepare(now);
            break;
        case "countdown":
            cursor.hide();
            tickCountdown(now);
            break;
        case "workout":
            cursor.hide();
            ({ bad, guides } = tickWorkout(now));
            break;
        default:
            cursor.hide();
    }
    draw(now, app.body, bad, guides);
}

// ---------- Меню ----------
function renderMenu() {
    const cards = $("#cards");
    cards.innerHTML = "";
    for (const def of EXERCISES) {
        const best = bestScore(def.id);
        const btn = document.createElement("button");
        btn.className = "card";
        btn.dataset.gesture = "";
        btn.innerHTML = `
            <span class="icon">${def.icon}</span>
            <h3>${def.name}</h3>
            <p>${def.desc}</p>
            <div class="meta"><span>${def.type === "hold" ? `${def.target} сек` : `${def.target} повторов`}</span>
            <span>${best ? `рекорд <b>${best}</b>` : "ещё не пробовал"}</span></div>`;
        btn.addEventListener("click", () => openPrepare(def));
        cards.appendChild(btn);
    }
    const h = loadHistory();
    const total = h.reduce((s, e) => s + (e.reps || 0), 0);
    $("#menu-stats").textContent = h.length
        ? `Всего подходов: ${h.length} · повторов: ${total} · Скрещённые руки над головой — закончить подход`
        : "Скрещённые руки над головой во время тренировки — закончить подход";
}

// ---------- Подготовка: тело в кадре и правильный ракурс ----------
function openPrepare(def) {
    app.def = def;
    app.prepView = null;
    app.prepOkSince = 0;
    $("#prep-eyebrow").textContent = def.type === "hold" ? `Цель: ${def.target} секунд с чистой техникой` : `Цель: ${def.target} повторов`;
    $("#prep-title").textContent = `${def.icon} ${def.name}`;
    $("#prep-setup").textContent = def.setup;
    $("#chk-view").textContent = { any: "Ракурс: лицом или боком", front: "Стоишь лицом к камере", side: "Стоишь боком к камере" }[def.view];
    $("#prep-checks").innerHTML = def.checks.map((c) => `<li>${c}</li>`).join("");
    $("#prep-fill").style.width = "0%";
    show("prepare");
    say(def.view === "side" ? "Встань боком к камере, чтобы я видел тебя целиком" : "Встань так, чтобы я видел тебя целиком", "prep", true);
}

function tickPrepare(now) {
    const def = app.def, b = app.body;
    const frameOk = !!b && inFrame(b, def.need);
    if (b) app.prepView = trackView(app.prepView, b);
    const viewOk = !!b && (def.view === "any" || app.prepView === def.view);
    $("#chk-frame").classList.toggle("ok", frameOk);
    $("#chk-view").classList.toggle("ok", viewOk);
    if (frameOk && viewOk) {
        app.prepOkSince ||= now;
        const p = clamp((now - app.prepOkSince) / 1200);
        $("#prep-fill").style.width = `${p * 100}%`;
        if (p >= 1) startCountdown(now);
    } else {
        app.prepOkSince = 0;
        $("#prep-fill").style.width = "0%";
    }
}

// ---------- Отсчёт 3-2-1 ----------
function startCountdown(now) {
    app.cd = { start: now, last: 0 };
    show("countdown");
}
function tickCountdown(now) {
    const t = now - app.cd.start;
    const n = 3 - Math.floor(t / 1000);
    if (t >= 3000) return startWorkout();
    if (n !== app.cd.last) {
        app.cd.last = n;
        const el = $("#count");
        el.textContent = n;
        el.classList.remove("pulse");
        void el.offsetWidth;
        el.classList.add("pulse");
        sfx.tick();
        say(String(n), "cd" + n, true);
    }
}

// ---------- Тренировка ----------
function startWorkout() {
    const def = app.def;
    app.ex = def.create();
    app.monitor = new FormMonitor();
    app.session = {
        def, start: performance.now(), lastT: 0,
        attempts: 0, counted: 0, clean: 0,
        repIssues: new Map(), stats: {},
        hold: 0, cleanHold: 0, everIn: false, outSince: 0,
        finishSince: 0, finishing: false,
    };
    $("#hud-type").textContent = def.type === "hold" ? "удержание" : "повторения";
    $("#hud-name").textContent = def.name;
    $("#hud-target").textContent = def.type === "hold" ? `/${def.target} с` : `/${def.target}`;
    show("workout");
    sfx.go();
    say(def.type === "hold" ? "Время пошло! Считаю только ровную планку" : "Начали!", "go", true);
}

const stat = (is) => (app.session.stats[is.code] ||= { issue: is, count: 0, time: 0 });

function tickWorkout(now) {
    const S = app.session, def = S.def, b = app.body;
    const dt = S.lastT ? Math.min(0.1, (now - S.lastT) / 1000) : 0;
    S.lastT = now;

    let res = null, issues;
    if (!b || !inFrame(b, def.need)) issues = [issue("not_visible")];
    else { res = app.ex.update(b, now); issues = res.issues; }

    const mon = app.monitor.update(issues, now);
    for (const is of mon.newly) {
        say(is.say || is.title, is.code);
        if (is.kind === "error") { sfx.bad(); if (def.type === "hold") stat(is).count++; }
    }
    const liveErrors = mon.active.filter((i) => i.kind === "error");

    if (def.type === "reps") {
        liveErrors.forEach((i) => S.repIssues.set(i.code, i));
        if (res?.rep) onRep(res.rep, now);
    } else {
        liveErrors.forEach((i) => { stat(i).time += dt; });
        if (res?.inPosition) {
            S.hold += dt;
            S.everIn = true;
            S.outSince = 0;
            if (!liveErrors.length) {
                const before = Math.floor(S.cleanHold);
                S.cleanHold += dt;
                const after = Math.floor(S.cleanHold);
                if (after !== before && after % 10 === 0 && after < def.target) say(`${after} секунд`, "sec" + after, true);
            }
        } else if (S.everIn) {
            S.outSince ||= now;
            if (now - S.outSince > 4000) finish("stopped"); // встал из планки
        }
        if (S.cleanHold >= def.target) finish("target");
    }

    // Жест завершения: скрещённые руки над головой
    const hint = $("#finish-hint");
    if (b && armsCrossedOverhead(b)) {
        S.finishSince ||= now;
        const p = clamp((now - S.finishSince) / FINISH_HOLD_MS);
        hint.style.setProperty("--p", p);
        hint.classList.add("active");
        if (p >= 1) finish("gesture");
    } else {
        S.finishSince = 0;
        hint.style.setProperty("--p", 0);
        hint.classList.remove("active");
    }

    if (app.screen !== "workout") return { bad: new Set(), guides: [] };
    updateHud(now, res, mon);

    const bad = new Set();
    for (const i of [...mon.active, mon.current].filter(Boolean)) (i.joints || []).forEach((j) => bad.add(j));
    return { bad, guides: res?.guides || [] };
}

function onRep(rep, now) {
    const S = app.session;
    S.attempts++;
    const errs = new Map(S.repIssues);
    S.repIssues.clear();
    rep.errors.forEach((e) => errs.set(e.code, e));
    errs.forEach((e) => { if (e.kind === "error") stat(e).count++; });

    if (rep.counted) {
        S.counted++;
        if (errs.size === 0) { S.clean++; app.goodUntil = now + 350; popRep("+1", "good"); }
        else popRep("+1", "warn");
        sfx.rep();
    } else {
        popRep("не засчитано", "bad");
        sfx.bad();
    }
    if (rep.errors.length) {
        const e = rep.errors[0];
        app.monitor.flash(e, 2600, now);
        say(e.say || e.title, e.code);
    }
    if (S.counted >= S.def.target && !S.finishing) {
        S.finishing = true;
        setTimeout(() => finish("target"), 500);
    }
}

function popRep(text, cls) {
    const el = $("#rep-pop");
    el.textContent = text;
    el.className = `rep-pop ${cls}`;
    void el.offsetWidth;
    el.classList.add("show");
}

const fmtTime = (sec) => `${Math.floor(sec / 60)}:${String(Math.floor(sec % 60)).padStart(2, "0")}`;

function updateHud(now, res, mon) {
    const S = app.session, def = S.def;
    $("#hud-time").textContent = fmtTime((now - S.start) / 1000);
    if (def.type === "reps") {
        $("#hud-count").textContent = S.counted;
        $("#hud-sub").textContent = S.attempts ? `чистых: ${S.clean} из ${S.attempts}` : "начинай, я считаю";
    } else {
        $("#hud-count").textContent = Math.floor(S.cleanHold);
        $("#hud-sub").textContent = S.hold > 0.5 ? `в планке всего: ${Math.floor(S.hold)} с` : "прими упор — таймер стартует сам";
    }
    $("#meter-fill").style.height = `${Math.round((res?.progress ?? 0) * 100)}%`;
    $("#meter-label").textContent = res?.metric?.label ?? "";
    $("#meter-value").textContent = res?.metric?.value ?? "";

    const box = $("#issue"), cur = mon.current;
    if (cur) {
        box.className = `issue ${cur.kind}`;
        $("#issue-icon").textContent = cur.kind === "error" ? "!" : cur.kind === "warn" ? "⚠" : "i";
        $("#issue-title").textContent = cur.title;
        $("#issue-fix").textContent = cur.fix;
    } else {
        box.className = "issue ok";
        $("#issue-icon").textContent = "✓";
        $("#issue-title").textContent = "Техника в норме";
        $("#issue-fix").textContent = def.type === "hold" ? "Держи линию тела и дыши ровно" : "Продолжай в том же темпе";
    }

    if (app.debug) {
        debugEl.textContent = JSON.stringify({ fps: app.fps.value, ...(res?.debug || {}), issues: mon.active.map((i) => i.code) }, null, 1);
    }
}

// ---------- Итоги ----------
function finish(reason) {
    if (app.screen !== "workout") return;
    const S = app.session, def = S.def;
    const duration = (performance.now() - S.start) / 1000;

    let score, quality;
    if (def.type === "reps") {
        score = S.counted * 10 + S.clean * 5;
        quality = S.attempts ? S.clean / S.attempts : 0;
    } else {
        score = Math.round(S.cleanHold * 3 + (S.hold - S.cleanHold));
        quality = S.hold ? S.cleanHold / S.hold : 0;
    }
    const empty = def.type === "reps" ? S.attempts === 0 : S.hold < 1;
    const prevBest = bestScore(def.id);
    const isRecord = !empty && score > 0 && score > prevBest;
    const entry = {
        ts: Date.now(), ex: def.id, score, quality: Math.round(quality * 100),
        reps: S.counted, attempts: S.attempts, clean: S.clean,
        hold: Math.round(S.hold), cleanHold: Math.round(S.cleanHold), duration: Math.round(duration), reason,
    };
    if (!empty) saveEntry(entry);

    // Заголовок и главная ошибка
    const errors = Object.values(S.stats).filter((s) => s.issue.kind === "error")
        .sort((a, b) => (def.type === "hold" ? b.time - a.time : b.count - a.count));
    const q = entry.quality;
    const headline = empty ? "Подход не начат"
        : q >= 80 ? "Отличная техника! 💪"
        : q >= 50 ? "Хорошо, но есть над чем поработать"
        : "Техника требует внимания";

    $("#sum-ex").textContent = `${def.icon} ${def.name} · ${fmtTime(duration)}`;
    $("#sum-headline").textContent = headline;
    $("#sum-score").textContent = score;
    $("#sum-record").hidden = !isRecord;

    const stats = def.type === "reps"
        ? [["Засчитано", `${S.counted}/${def.target}`], ["Чистых", S.clean], ["Попыток", S.attempts], ["Техника", `${q}%`]]
        : [["Чистое время", `${Math.round(S.cleanHold)} с`], ["Всего в планке", `${Math.round(S.hold)} с`], ["Цель", `${def.target} с`], ["Техника", `${q}%`]];
    $("#sum-stats").innerHTML = stats.map(([k, v]) => `<div class="stat"><small>${k}</small><b>${v}</b></div>`).join("");

    $("#sum-errors").innerHTML = errors.length
        ? errors.slice(0, 4).map((s) => `
            <li><b>${s.issue.title}<em>${def.type === "hold" ? `${Math.round(s.time)} с` : `×${s.count}`}</em></b>
            <span>Как исправить: ${s.issue.fix}</span></li>`).join("")
        : `<li class="clean"><b>Ошибок не замечено</b><span>${empty ? "Сделай хотя бы одно движение — и я разберу технику." : "Так держать! Попробуй увеличить темп или цель."}</span></li>`;

    const hist = loadHistory().filter((e) => e.ex === def.id).slice(-8);
    const max = Math.max(1, ...hist.map((e) => e.score));
    $("#sum-bars").innerHTML = hist.length
        ? hist.map((e, i) => `<div class="${i === hist.length - 1 && !empty ? "last" : ""}" style="height:${Math.max(4, (e.score / max) * 100)}%"><span>${e.score}</span></div>`).join("")
        : `<p class="bars-empty">Здесь появится график после первых подходов</p>`;

    show("summary");
    sfx.finish();
    const top = errors[0];
    say(empty ? "Подход завершён" : `Готово! ${score} очков. ${isRecord ? "Новый рекорд! " : ""}${top ? "Главное замечание: " + top.issue.title : "Отличная техника!"}`, "finish", true);
}

/* =====================================================================
   11. Кнопки и клавиатура (запасное управление)
   ===================================================================== */
$("#btn-start").addEventListener("click", () => start({ camera: true }));
$("#file-input").addEventListener("change", (e) => {
    const f = e.target.files[0];
    if (f) start({ file: f });
});
$("#btn-retry").addEventListener("click", () => start({ camera: true }));
$("#btn-prep-back").addEventListener("click", () => { renderMenu(); show("menu"); });
$("#btn-stop").addEventListener("click", () => finish("button"));
$("#btn-again").addEventListener("click", () => openPrepare(app.def));
$("#btn-menu").addEventListener("click", () => { renderMenu(); show("menu"); });

const muteBtn = $("#btn-mute");
function toggleMute() {
    muted = !muted;
    muteBtn.textContent = muted ? "🔇" : "🔊";
    if (muted && "speechSynthesis" in window) speechSynthesis.cancel();
}
muteBtn.addEventListener("click", toggleMute);

window.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
        if (app.screen === "workout") finish("key");
        else if (app.screen === "prepare" || app.screen === "countdown") { renderMenu(); show("menu"); }
    } else if (e.key === "m" || e.key === "M" || e.key === "ь" || e.key === "Ь") {
        toggleMute();
    } else if (e.key === "d" || e.key === "D" || e.key === "в" || e.key === "В") {
        app.debug = !app.debug;
        debugEl.hidden = !app.debug;
    }
});

window.addEventListener("resize", updateFit);
video.addEventListener("loadedmetadata", updateFit);
if ("speechSynthesis" in window) speechSynthesis.getVoices(); // прогреваем список голосов

// Скрипт загрузился — кнопки работают
window.__formaReady = true;
setStatus("Нажми «Включить камеру»");

// Если доступ к камере уже разрешён раньше — включаем её сразу, без кнопки
navigator.permissions?.query({ name: "camera" })
    .then((p) => { if (p.state === "granted" && app.screen === "intro") start({ camera: true }); })
    .catch(() => { /* браузер не поддерживает проверку разрешений */ });