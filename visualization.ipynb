import React, { useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Line,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const priceData = [
  { day: "D-19", close: 68400, ma20: 67600, volume: 0.72, regime: 1 },
  { day: "D-18", close: 68900, ma20: 67850, volume: 0.81, regime: 1 },
  { day: "D-17", close: 68100, ma20: 67920, volume: 0.78, regime: 1 },
  { day: "D-16", close: 69500, ma20: 68110, volume: 1.02, regime: 1 },
  { day: "D-15", close: 70100, ma20: 68340, volume: 1.11, regime: 1 },
  { day: "D-14", close: 69800, ma20: 68520, volume: 0.95, regime: 1 },
  { day: "D-13", close: 70600, ma20: 68790, volume: 1.24, regime: 2 },
  { day: "D-12", close: 71200, ma20: 69060, volume: 1.32, regime: 2 },
  { day: "D-11", close: 70900, ma20: 69240, volume: 1.08, regime: 2 },
  { day: "D-10", close: 71600, ma20: 69510, volume: 1.35, regime: 2 },
  { day: "D-9", close: 72100, ma20: 69800, volume: 1.41, regime: 2 },
  { day: "D-8", close: 71800, ma20: 70020, volume: 1.16, regime: 2 },
  { day: "D-7", close: 72500, ma20: 70300, volume: 1.28, regime: 2 },
  { day: "D-6", close: 73100, ma20: 70660, volume: 1.53, regime: 2 },
  { day: "D-5", close: 72800, ma20: 70940, volume: 1.19, regime: 2 },
  { day: "D-4", close: 73400, ma20: 71200, volume: 1.34, regime: 2 },
  { day: "D-3", close: 73900, ma20: 71560, volume: 1.48, regime: 2 },
  { day: "D-2", close: 73600, ma20: 71820, volume: 1.22, regime: 2 },
  { day: "D-1", close: 74200, ma20: 72100, volume: 1.51, regime: 2 },
  { day: "Today", close: 74600, ma20: 72420, volume: 1.56, regime: 2 },
];

const featureImportance = [
  { name: "수급 신호", value: 31.2, corr: 0.08 },
  { name: "모멘텀", value: 24.7, corr: 0.06 },
  { name: "RSI", value: 18.3, corr: 0.05 },
  { name: "거래량", value: 12.5, corr: 0.03 },
  { name: "추세", value: 8.8, corr: 0.04 },
  { name: "저변동성", value: 4.5, corr: 0.02 },
];

const signalRows = [
  { signal: "supply_signal", value: 0.78, corr: 0.08, usable: true, desc: "외국인 5일 누적 순매수 분위수" },
  { signal: "momentum", value: 0.71, corr: 0.06, usable: true, desc: "20일 수익률 분위수" },
  { signal: "rsi_norm", value: 0.44, corr: 0.05, usable: true, desc: "RSI 14 정규화" },
  { signal: "trend", value: 0.58, corr: 0.04, usable: true, desc: "MA20 대비 현재가 위치" },
  { signal: "volume_signal", value: 0.63, corr: 0.03, usable: true, desc: "거래량 20일 평균 대비 배율" },
  { signal: "low_vol", value: 0.52, corr: 0.02, usable: false, desc: "20일 변동성 역수" },
];

const backtestData = [
  { period: "Q1", ret: 5.8, win: 58 },
  { period: "Q2", ret: 7.2, win: 62 },
  { period: "Q3", ret: 9.1, win: 66 },
  { period: "Q4", ret: 6.7, win: 61 },
];

const regimeDistribution = [
  { name: "강세", value: 38 },
  { name: "횡보", value: 41 },
  { name: "약세", value: 21 },
];

const topReasons = [
  {
    rank: 1,
    title: "외국인 수급 강함",
    sign: "+",
    value: "상관계수 0.08",
    detail: "최근 수급 신호가 가장 높은 기여도를 보이며, 외국인 누적 순매수 분위수가 상위권입니다.",
  },
  {
    rank: 2,
    title: "모멘텀 상승 중",
    sign: "+",
    value: "20일 모멘텀 0.71",
    detail: "가격이 MA20 위에서 유지되며 단기 추세가 우호적으로 전환되고 있습니다.",
  },
  {
    rank: 3,
    title: "RSI 중립",
    sign: "LOW RISK",
    value: "RSI 44.2",
    detail: "과열 구간이 아니므로 추격 매수 위험이 상대적으로 낮습니다.",
  },
];

const riskNarratives = [
  { label: "내가 감수해야 할 하락폭", value: "-3.1%", desc: "현재가 기준 손절선까지의 거리입니다." },
  { label: "기대할 수 있는 상승 여지", value: "+7.0%", desc: "목표가까지 도달했을 때의 예상 수익 구간입니다." },
  { label: "수익 대비 위험", value: "2배", desc: "손실 1을 감수할 때 수익 2를 기대하는 구조입니다." },
  { label: "권고 진입 비중", value: "4.2%", desc: "Kelly 기준으로 과도한 집중을 피한 보수적 비중입니다." },
];

const ICONS = {
  activity: "↗",
  alert: "!",
  bar: "▦",
  brain: "ML",
  check: "✓",
  database: "DB",
  line: "⌁",
  shield: "◇",
  trend: "▲",
  upload: "↑",
};

// 🔹 용어 설명 (툴팁용)
const TERM_EXPLANATIONS = {
  "품질 점수": "데이터가 얼마나 신뢰할 수 있는지를 평가한 점수",
  "레짐": "현재 시장 상태 (강세, 횡보, 약세)",
  "Kelly": "리스크를 고려해 계산된 최적 투자 비중",
  "CV R²": "모델이 과거 검증 구간에서 설명력을 얼마나 가지는지",
  "Alpha Score": "여러 신호를 종합한 투자 매력도 점수",
  "Walk-forward": "과거로 학습하고 미래 구간으로 검증하는 백테스트 방식",
};

function scoreAction(score) {
  if (score >= 80) return { label: "Strong Buy", tone: "text-emerald-600", bg: "bg-emerald-50 border-emerald-200" };
  if (score >= 60) return { label: "Buy", tone: "text-blue-600", bg: "bg-blue-50 border-blue-200" };
  if (score >= 40) return { label: "Neutral", tone: "text-amber-600", bg: "bg-amber-50 border-amber-200" };
  if (score >= 20) return { label: "Caution", tone: "text-orange-600", bg: "bg-orange-50 border-orange-200" };
  return { label: "Avoid", tone: "text-red-600", bg: "bg-red-50 border-red-200" };
}

function getPipelineProgress(stage) {
  if (stage === "SCANNING") return 35;
  if (stage === "MODELING") return 72;
  if (stage === "COMPLETE") return 100;
  return 0;
}

function runDashboardSelfTests() {
  const cases = [
    [95, "Strong Buy"],
    [80, "Strong Buy"],
    [74.2, "Buy"],
    [60, "Buy"],
    [50, "Neutral"],
    [25, "Caution"],
    [10, "Avoid"],
  ];

  cases.forEach(([score, expected]) => {
    const actual = scoreAction(score).label;
    console.assert(actual === expected, `scoreAction(${score}) expected ${expected}, got ${actual}`);
  });

  console.assert(getPipelineProgress("READY") === 0, "READY stage should be 0%.");
  console.assert(getPipelineProgress("SCANNING") === 35, "SCANNING stage should be 35%.");
  console.assert(getPipelineProgress("MODELING") === 72, "MODELING stage should be 72%.");
  console.assert(getPipelineProgress("COMPLETE") === 100, "COMPLETE stage should be 100%.");
  console.assert(priceData.length >= 20, "priceData should include at least 20 rows for the demo chart.");
  console.assert(signalRows.some((row) => row.usable === false), "At least one signal should demonstrate low reliability handling.");
  console.assert(topReasons.length === 3, "Top reasons should contain exactly three items.");
  console.assert(riskNarratives.length === 4, "Risk narratives should contain four human-readable cards.");
}

if (typeof window !== "undefined") {
  runDashboardSelfTests();
}

function InfoTooltip({ term }) {
  return (
    <span className="relative group ml-1 cursor-pointer text-slate-400">
      (?)
      <span className="absolute left-1/2 top-6 z-50 hidden w-56 -translate-x-1/2 rounded-xl bg-slate-900 p-2 text-xs text-white group-hover:block">
        {TERM_EXPLANATIONS[term]}
      </span>
    </span>
  );
}

function IconBadge({ type = "activity", className = "" }) {
  return (
    <span
      className={`inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-2xl bg-slate-100 text-xs font-black text-slate-700 ${className}`}
      aria-hidden="true"
    >
      {ICONS[type] || "•"}
    </span>
  );
}

function Card({ children, className = "" }) {
  return <div className={`rounded-2xl border border-slate-200 bg-white p-5 shadow-sm ${className}`}>{children}</div>;
}

function SectionTitle({ icon = "activity", title, subtitle }) {
  return (
    <div className="mb-4 flex items-start justify-between gap-4">
      <div className="flex items-center gap-3">
        <IconBadge type={icon} />
        <div>
          <h2 className="text-lg font-bold text-slate-900">{title}</h2>
          {subtitle && <p className="text-sm text-slate-500">{subtitle}</p>}
        </div>
      </div>
    </div>
  );
}

function ScoreGauge({ score }) {
  const action = scoreAction(score);
  return (
    <div className="flex flex-col items-center justify-center">
      <div className="relative h-44 w-44 rounded-full bg-slate-100">
        <div
          className="absolute inset-0 rounded-full"
          style={{ background: `conic-gradient(#2563eb ${score * 3.6}deg, #e2e8f0 0deg)` }}
        />
        <div className="absolute inset-4 flex flex-col items-center justify-center rounded-full bg-white shadow-inner">
          <span className="text-4xl font-black text-slate-900">{score}</span>
          <span className="text-xs font-semibold text-slate-500">Alpha Score</span>
        </div>
      </div>
      <div className={`mt-4 rounded-full border px-4 py-2 text-sm font-bold ${action.bg} ${action.tone}`}>{action.label}</div>
    </div>
  );
}

function StatCard({ icon, label, value, sub }) {
  return (
    <Card>
      <div className="flex items-center justify-between">
        <IconBadge type={icon} className="h-8 w-8" />
        <span className="rounded-full bg-slate-100 px-2 py-1 text-xs font-semibold text-slate-500">자동 계산</span>
      </div>
      <p className="mt-4 text-sm text-slate-500">{label}</p>
      <p className="mt-1 text-2xl font-black text-slate-900">{value}</p>
      <p className="mt-1 text-xs text-slate-500">{sub}</p>
    </Card>
  );
}

export default function AlphaSignalDashboard() {
  const [ticker, setTicker] = useState("005930.KS");
  const [uploaded, setUploaded] = useState(false);
  const [analysisStage, setAnalysisStage] = useState("READY");
  const [showJson, setShowJson] = useState(false);

  const score = 74.2;
  const latest = priceData[priceData.length - 1];
  const action = scoreAction(score);
  const progress = getPipelineProgress(analysisStage);

  function runAnalysis() {
    setUploaded(true);
    setAnalysisStage("SCANNING");
    window.setTimeout(() => setAnalysisStage("MODELING"), 450);
    window.setTimeout(() => setAnalysisStage("COMPLETE"), 900);
  }

  const jsonOutput = useMemo(
    () => ({
      meta: {
        ticker: ticker.replace(".KS", ""),
        name: ticker.includes("005930") ? "삼성전자" : "사용자 입력 종목",
        data_types: ["price", "ohlcv", "supply", "technical"],
        data_period_days: 252,
        quality: { total: 88, grade: "신뢰 높음" },
      },
      regime: { current: 2, label: "강세 (Bullish)" },
      alpha_score: { score, action: action.label, cv_r2: 0.042, reliability: "보통" },
      risk: { stop_loss: 71120, target_price: 78540, kelly_recommended_pct: "4.2%" },
      signals: signalRows.reduce((acc, row) => {
        acc[row.signal] = { value: row.value, corr: row.corr, usable: row.usable };
        return acc;
      }, {}),
      insight: {
        top_signals: ["수급 신호 강함", "모멘텀 양호", "RSI 중립"],
        final_decision: "BUY · 제한 진입",
        disclaimer: "과거 성과가 미래를 보장하지 않습니다.",
      },
    }),
    [ticker, action.label]
  );

  return (
    <div className="min-h-screen bg-slate-50 px-6 py-8 text-slate-900">
      <div className="mx-auto max-w-7xl">
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="mb-6 overflow-hidden rounded-3xl bg-slate-950 p-7 text-white shadow-lg">
          <div className="flex flex-col justify-between gap-5 lg:flex-row lg:items-end">
            <div>
              <div className="mb-3 inline-flex items-center gap-2 rounded-full bg-white/10 px-3 py-1 text-sm text-slate-200">
                <span className="inline-flex h-5 w-5 items-center justify-center rounded-full bg-white/15 text-[10px] font-black">ML</span>
                K-Means + Ridge + Walk-forward
              </div>
              <h1 className="text-3xl font-black tracking-tight lg:text-5xl">Alpha Signal Dashboard</h1>
              <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-300">
                Skills.md v3.0 기반 ML 투자 분석 대시보드. 데이터 유형을 자동 감지하고, 레짐·Alpha Score·리스크·백테스트·인사이트를 한 화면에 생성합니다.
              </p>
            </div>
            <div className="flex flex-col gap-3 rounded-2xl bg-white/10 p-4 backdrop-blur lg:w-[360px]">
              <label className="text-xs font-semibold text-slate-300">Ticker / File Input</label>
              <div className="flex gap-2">
                <input
                  value={ticker}
                  onChange={(event) => setTicker(event.target.value)}
                  className="min-w-0 flex-1 rounded-xl border border-white/10 bg-white px-3 py-2 text-sm text-slate-900 outline-none"
                  aria-label="ticker input"
                />
                <button onClick={runAnalysis} className="rounded-xl bg-blue-500 px-4 py-2 text-sm font-bold text-white hover:bg-blue-400">
                  Analyze
                </button>
              </div>
              <button onClick={runAnalysis} className="flex items-center justify-center gap-2 rounded-xl border border-white/20 px-3 py-2 text-sm text-slate-200 hover:bg-white/10">
                <span className="font-black">↑</span> CSV/XLSX 업로드 데모
              </button>
              {uploaded && (
                <div className="rounded-xl border border-white/10 bg-slate-950/60 p-3">
                  <div className="mb-2 flex items-center justify-between text-xs text-slate-300">
                    <span>System Pipeline</span>
                    <span className="font-bold text-emerald-300">{analysisStage}</span>
                  </div>
                  <div className="h-2 overflow-hidden rounded-full bg-white/10">
                    <motion.div
                      className="h-full rounded-full bg-gradient-to-r from-blue-400 via-indigo-300 to-emerald-300"
                      initial={{ width: "0%" }}
                      animate={{ width: `${progress}%` }}
                      transition={{ duration: 0.45 }}
                    />
                  </div>
                  <p className="mt-2 text-xs text-emerald-300">입력 데이터 감지 완료 · price / ohlcv / supply / technical</p>
                </div>
              )}
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.08 }}
          className="mb-6 overflow-hidden rounded-3xl border border-slate-800 bg-gradient-to-br from-slate-950 via-slate-900 to-blue-950 p-6 text-white shadow-2xl"
        >
          <div className="grid gap-6 lg:grid-cols-12 lg:items-center">
            <div className="lg:col-span-5">
              <p className="text-xs font-black uppercase tracking-[0.35em] text-blue-200">Final Decision</p>
              <div className="mt-3 flex flex-wrap items-end gap-3">
                <span className="text-5xl font-black tracking-tight text-white">BUY</span>
                <span className="mb-2 rounded-full border border-blue-300/40 bg-blue-400/15 px-4 py-1 text-sm font-bold text-blue-100">신뢰도: 보통</span>
              </div>
              <p className="mt-4 max-w-xl text-sm leading-7 text-slate-300">
                현재 분석 결과는 <b className="text-white">공격적 매수</b>보다는, 우호적인 신호를 확인하고 <b className="text-white">포트폴리오의 4.2% 수준으로 제한 진입</b>하는 판단에 가깝습니다.
              </p>
            </div>
            <div className="grid gap-3 sm:grid-cols-3 lg:col-span-7">
              {topReasons.map((reason) => (
                <motion.div
                  key={reason.rank}
                  whileHover={{ y: -4, scale: 1.01 }}
                  className="rounded-2xl border border-white/10 bg-white/10 p-4 backdrop-blur"
                >
                  <div className="flex items-center justify-between">
                    <span className="flex h-8 w-8 items-center justify-center rounded-xl bg-white text-sm font-black text-slate-950">{reason.rank}</span>
                    <span className="rounded-full bg-emerald-400/15 px-2 py-1 text-xs font-black text-emerald-200">{reason.sign}</span>
                  </div>
                  <h3 className="mt-4 text-base font-black text-white">{reason.title}</h3>
                  <p className="mt-1 text-xs font-bold text-blue-200">{reason.value}</p>
                  <p className="mt-3 text-xs leading-5 text-slate-300">{reason.detail}</p>
                </motion.div>
              ))}
            </div>
          </div>
        </motion.div>

        <div className="mb-6 grid grid-cols-2 gap-4 lg:grid-cols-5">
          <StatCard icon="database" label={<>품질 점수 <InfoTooltip term="품질 점수" /></>} value="88점" sub="신뢰 높음" />
          <StatCard icon="activity" label={<>현재 레짐 <InfoTooltip term="레짐" /></>} value="강세" sub="K-Means" />
          <StatCard icon="trend" label="현재가" value={latest.close.toLocaleString()} sub="+0.54%" />
          <StatCard icon="shield" label={<>Kelly 비중 <InfoTooltip term="Kelly" /></>} value="4.2%" sub="Quarter Kelly" />
          <StatCard icon="alert" label={<>CV R² <InfoTooltip term="CV R²" /></>} value="0.042" sub="보통 신뢰도" />
        </div>

        <div className="grid gap-6 lg:grid-cols-12">
          <Card className="lg:col-span-4">
            <SectionTitle icon="bar" title={<>Alpha Score <InfoTooltip term="Alpha Score" /></>} subtitle="Ridge 예측값을 분위수 기반 0~100점으로 변환" />
            <ScoreGauge score={score} />
            <div className="mt-5 rounded-2xl border border-blue-100 bg-blue-50 p-4">
              <p className="text-sm font-bold text-blue-700">최종 판단: {action.label}</p>
              <p className="mt-2 text-sm leading-6 text-blue-900">점수만 보면 매수 우위지만, 모델 신뢰도가 중간 수준이므로 전액 진입이 아니라 제한 진입이 적절합니다.</p>
            </div>
          </Card>

          <Card className="lg:col-span-8">
            <SectionTitle icon="line" title="가격 + MA20 + 거래량" subtitle="감지 유형: price + ohlcv → 메인 차트 자동 선택" />
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={priceData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="day" tick={{ fontSize: 12 }} />
                  <YAxis yAxisId="left" domain={[66000, 76000]} tick={{ fontSize: 12 }} />
                  <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 12 }} />
                  <Tooltip />
                  <Bar yAxisId="right" dataKey="volume" barSize={16} opacity={0.25} name="거래량 배율" />
                  <Line yAxisId="left" type="monotone" dataKey="close" strokeWidth={3} dot={false} name="Close" />
                  <Line yAxisId="left" type="monotone" dataKey="ma20" strokeWidth={2} dot={false} name="MA20" />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
            <div className="mt-3 grid gap-3 md:grid-cols-3">
              <div className="rounded-xl bg-emerald-50 p-3 text-sm"><b>목표가</b><br />78,540원 (+7.0%)</div>
              <div className="rounded-xl bg-red-50 p-3 text-sm"><b>손절가</b><br />71,120원 (-3.1%)</div>
              <div className="rounded-xl bg-slate-100 p-3 text-sm"><b>RR 비율</b><br />2.0 : 1</div>
            </div>
          </Card>

          <Card className="lg:col-span-6">
            <SectionTitle icon="brain" title="피처 기여도" subtitle="Ridge 계수 기반 모델 투명성 패널" />
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={featureImportance} layout="vertical" margin={{ left: 24 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" tick={{ fontSize: 12 }} />
                  <YAxis dataKey="name" type="category" tick={{ fontSize: 12 }} width={80} />
                  <Tooltip />
                  <Bar dataKey="value" name="중요도(%)" radius={[0, 10, 10, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Card>

          <Card className="lg:col-span-6">
            <SectionTitle icon="activity" title="레짐 분포" subtitle="K-Means: 수익률 + 변동성 2차원 클러스터" />
            <div className="grid items-center gap-5 md:grid-cols-2">
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie data={regimeDistribution} innerRadius={58} outerRadius={92} dataKey="value" label>
                      {regimeDistribution.map((entry) => (
                        <Cell key={entry.name} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div className="space-y-3 text-sm">
                <div className="rounded-xl bg-emerald-50 p-3"><b>강세</b> 평균 수익률 4.8% · 빈도 38%</div>
                <div className="rounded-xl bg-slate-100 p-3"><b>횡보</b> 평균 수익률 0.2% · 빈도 41%</div>
                <div className="rounded-xl bg-red-50 p-3"><b>약세</b> 평균 수익률 -3.1% · 빈도 21%</div>
              </div>
            </div>
          </Card>

          <Card className="lg:col-span-7">
            <SectionTitle icon="check" title="Rolling 상관계수 신호 검증" subtitle="각 피처와 20일 미래수익률의 최근 상관계수" />
            <div className="overflow-hidden rounded-2xl border border-slate-200">
              <table className="w-full text-left text-sm">
                <thead className="bg-slate-100 text-slate-600">
                  <tr>
                    <th className="px-4 py-3">Signal</th>
                    <th className="px-4 py-3">설명</th>
                    <th className="px-4 py-3">값</th>
                    <th className="px-4 py-3">Corr</th>
                    <th className="px-4 py-3">상태</th>
                  </tr>
                </thead>
                <tbody>
                  {signalRows.map((row) => (
                    <tr key={row.signal} className="border-t border-slate-100">
                      <td className="px-4 py-3 font-semibold">{row.signal}</td>
                      <td className="px-4 py-3 text-slate-500">{row.desc}</td>
                      <td className="px-4 py-3">{row.value}</td>
                      <td className="px-4 py-3">{row.corr}</td>
                      <td className="px-4 py-3">
                        <span className={`rounded-full px-2 py-1 text-xs font-bold ${row.usable ? "bg-emerald-50 text-emerald-700" : "bg-slate-100 text-slate-500"}`}>
                          {row.usable ? "사용" : "낮은 신뢰"}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card className="lg:col-span-5">
            <SectionTitle icon="bar" title="Walk-forward 백테스트" subtitle="1년 학습 → 3개월 검증 반복" />
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={backtestData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="period" tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 12 }} />
                  <Tooltip />
                  <Bar dataKey="ret" name="평균 수익률(%)" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="grid grid-cols-3 gap-2 text-center text-sm">
              <div className="rounded-xl bg-slate-100 p-3"><b>7.2%</b><br />평균</div>
              <div className="rounded-xl bg-slate-100 p-3"><b>62%</b><br />승률</div>
              <div className="rounded-xl bg-slate-100 p-3"><b>1.24</b><br />Sharpe</div>
            </div>
          </Card>

          <Card className="lg:col-span-6">
            <SectionTitle icon="shield" title="자연어 인사이트" subtitle="최종 판단과 근거를 사용자 언어로 요약" />
            <div className="overflow-hidden rounded-2xl bg-slate-950 text-slate-100">
              <div className="border-b border-white/10 bg-gradient-to-r from-blue-950 to-slate-950 p-5">
                <p className="text-xs font-black uppercase tracking-[0.3em] text-blue-200">Investment Verdict</p>
                <div className="mt-3 flex items-center justify-between gap-4">
                  <div>
                    <p className="text-3xl font-black text-white">BUY · 제한 진입</p>
                    <p className="mt-2 text-sm leading-6 text-slate-300"><b>{ticker}</b>는 현재 Alpha Score <b>{score}점</b>으로 매수 우위입니다.</p>
                  </div>
                  <div className="rounded-2xl border border-blue-300/20 bg-blue-400/10 px-4 py-3 text-center">
                    <p className="text-xs text-blue-200">권고 비중</p>
                    <p className="text-2xl font-black text-white">4.2%</p>
                  </div>
                </div>
              </div>

              <div className="grid gap-3 p-5">
                {topReasons.map((reason) => (
                  <div key={reason.rank} className="flex gap-3 rounded-2xl border border-white/10 bg-white/5 p-4">
                    <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-white text-sm font-black text-slate-950">{reason.rank}</span>
                    <div>
                      <p className="font-black text-white">{reason.title} <span className="text-emerald-300">({reason.sign})</span></p>
                      <p className="mt-1 text-xs font-bold text-blue-200">{reason.value}</p>
                      <p className="mt-2 text-xs leading-5 text-slate-300">{reason.detail}</p>
                    </div>
                  </div>
                ))}
              </div>
              <p className="border-t border-white/10 p-5 text-sm leading-7 text-amber-200">※ 과거 성과가 미래를 보장하지 않습니다. 이 시스템은 투자 보조 도구입니다.</p>
            </div>
          </Card>

          <Card className="lg:col-span-6">
            <SectionTitle icon="shield" title="리스크 관리" subtitle="숫자를 실제 의사결정 언어로 변환" />
            <div className="grid gap-3 sm:grid-cols-2">
              {riskNarratives.map((item) => (
                <motion.div key={item.label} whileHover={{ y: -3 }} className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                  <p className="text-xs font-bold text-slate-500">{item.label}</p>
                  <p className="mt-2 text-3xl font-black text-slate-950">{item.value}</p>
                  <p className="mt-2 text-xs leading-5 text-slate-500">{item.desc}</p>
                </motion.div>
              ))}
            </div>
            <div className="mt-4 rounded-2xl border border-slate-800 bg-slate-950 p-4 text-sm leading-6 text-slate-200">
              이 전략은 <b className="text-white">손실을 -3.1%에서 제한</b>하고, 성공 시 <b className="text-white">+7.0% 구간</b>을 노리는 구조입니다. 즉, 한 번 틀렸을 때보다 한 번 맞았을 때의 보상이 약 2배 큽니다.
            </div>
            <div className="mt-4">
              <button onClick={() => setShowJson(!showJson)} className="rounded-xl border border-slate-300 px-4 py-2 text-sm font-bold text-slate-700 hover:bg-slate-100">
                {showJson ? "Developer JSON 숨기기" : "Developer JSON 보기"}
              </button>
              {showJson && (
                <pre className="mt-4 max-h-72 overflow-auto rounded-2xl bg-slate-950 p-4 text-xs leading-5 text-emerald-100">
                  {JSON.stringify(jsonOutput, null, 2)}
                </pre>
              )}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
