function(theme, span) {
    const S = window.exoFeed = window.exoFeed || {skip: 0, traces: 0, paused: false, gen: 0};
    Object.assign(S, {theme, span, cursor: null, shown: {}, gen: S.gen + 1});
    if (S.running) return null;
    S.running = true;
    const config = {displaylogo: false, responsive: true};
    const hold = on => {
        if (S.paused === on) return;
        S.paused = on;
        dash_clientside.set_props("live-hint", {children: on ? "held while zoomed · double-click to follow" : ""});
    };
    const draw = (gd, feed) => {
        if (feed.figure) {
            Plotly.react(gd, feed.figure.data, feed.figure.layout,
                         {...config, doubleClick: "autosize", modeBarButtonsToRemove: ["resetScale2d"]});
            Object.assign(S, {traces: feed.figure.data.length, skip: 0});
            hold(false);
            if (!gd.exoZoom) {
                gd.exoZoom = true;
                gd.on("plotly_relayout", e => {
                    const keys = Object.keys(e || {});
                    if (keys.some(k => k.endsWith("autorange") && e[k])) {
                        hold(false);
                        Object.assign(S, {cursor: null, gen: S.gen + 1});  // the whole figure again
                    } else if (keys.some(k => k.includes("range["))) hold(true);
                });
            }
            return feed.latest;
        }
        // a remounted graph lost its traces: ask for the whole figure again
        if (!gd.data || gd.data.length !== S.traces) return null;
        const keep = feed.t.map((_, i) => (S.skip + i) % STRIDE === 0);
        S.skip = (S.skip + feed.t.length) % STRIDE;
        const pick = a => a.filter((_, i) => keep[i]);
        const t = pick(feed.t), xs = [], ys = [];
        // phase lines break where the phase wraps
        feed.y.forEach((y, i) => {
            const x = [], v = [];
            let last = gd.data[i].y[gd.data[i].y.length - 1];
            pick(y).forEach((value, j) => {
                if (PHASE.includes(i) && value !== null && last !== null && value < last - 0.5) {
                    x.push(t[j]); v.push(null);
                }
                x.push(t[j]); v.push(value); last = value;
            });
            xs.push(x); ys.push(v);
        });
        // the new range rides along with the extension's redraw: one redraw per chunk
        Object.assign(gd.layout.xaxis, {range: [feed.latest - WINDOW, feed.latest], autorange: false});
        if (t.length) Plotly.extendTraces(gd, {x: xs, y: ys}, feed.y.map((_, i) => i), POINTS);
        return feed.latest;
    };
    // a request continues from the reply before it, so the next one is sent while this one is drawn
    const request = () => {
        const gen = S.gen;
        const query = new URLSearchParams({theme: S.theme || "", span: S.span || "window",
            paused: S.paused ? 1 : 0, cursor: S.cursor ?? "", ...S.shown});
        return fetch("live-feed?" + query).then(r => r.json()).then(feed => ({feed, gen}), () => ({gen: -1}));
    };
    const step = async () => {
        const started = performance.now();
        let wait = PERIOD;
        try {
            const gd = document.getElementById("live");
            if (!gd || !window.Plotly || gd.offsetParent === null) {
                wait = 500;  // not on the session page: redraw whole figures once it shows again
                Object.assign(S, {cursor: null, shown: {}, gen: S.gen + 1, next: null});
            } else {
                const {feed, gen} = await (S.next || request());
                S.next = null;
                if (!feed) {
                    wait = 500;
                } else if (gen === S.gen) {  // else the view or theme changed meanwhile
                    if (feed.traces) S.cursor = feed.traces.latest ?? null;
                    for (const [id, panel] of Object.entries(feed.panels || {})) S.shown[id] = panel.key;
                    if (!feed.idle) S.next = request();
                    if (feed.traces && draw(gd, feed.traces) === null) {
                        Object.assign(S, {cursor: null, gen: S.gen + 1});  // lost its traces: the whole figure
                    }
                    for (const [id, panel] of Object.entries(feed.panels || {})) {
                        const div = document.getElementById(id);
                        if (div) Plotly.react(div, panel.figure.data, panel.figure.layout, config);
                    }
                }
                if (feed && feed.idle) wait = 500;
            }
        } catch (error) {
            wait = 500;
        }
        setTimeout(step, Math.max(0, wait - (performance.now() - started)));
    };
    step();
    return null;
}
