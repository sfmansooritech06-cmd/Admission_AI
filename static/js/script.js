/* ════════════════════════════════════════════════════
   AdmitAI – Landing Page Scripts
════════════════════════════════════════════════════ */

"use strict";

/* ── Navbar scroll effect ─────────────────────────── */
(function () {
    const navbar = document.getElementById("navbar");
    if (!navbar) return;
    window.addEventListener("scroll", () => {
        if (window.scrollY > 20) {
            navbar.classList.add("scrolled");
        } else {
            navbar.classList.remove("scrolled");
        }
    });
})();

/* ── Mobile menu toggle ───────────────────────────── */
(function () {
    const toggle = document.getElementById("navToggle");
    const navbar = document.getElementById("navbar");
    if (!toggle || !navbar) return;

    toggle.addEventListener("click", () => {
        navbar.classList.toggle("menu-open");
    });

    // Close menu on link click
    document.querySelectorAll(".nav-link").forEach(link => {
        link.addEventListener("click", () => navbar.classList.remove("menu-open"));
    });
})();

/* ── Hero typing animation ───────────────────────── */
(function () {
    const el = document.getElementById("typingText");
    if (!el) return;

    const words = [
        "College",
        "Admission",
        "AI-Powered",
        "Smart",
        "Instant",
    ];
    let wordIndex = 0;
    let charIndex = 0;
    let isDeleting = false;
    const typingSpeed = 100;
    const deletingSpeed = 60;
    const pauseTime = 2200;

    function type() {
        const current = words[wordIndex];
        el.classList.add("typing-cursor");

        if (isDeleting) {
            el.textContent = current.substring(0, charIndex - 1);
            charIndex--;
        } else {
            el.textContent = current.substring(0, charIndex + 1);
            charIndex++;
        }

        if (!isDeleting && charIndex === current.length) {
            setTimeout(() => { isDeleting = true; type(); }, pauseTime);
            return;
        }
        if (isDeleting && charIndex === 0) {
            isDeleting = false;
            wordIndex = (wordIndex + 1) % words.length;
        }

        setTimeout(type, isDeleting ? deletingSpeed : typingSpeed);
    }
    setTimeout(type, 600);
})();

/* ── Counter animation ───────────────────────────── */
function animateCounter(el, target, suffix) {
    const duration = 2000;
    const start = performance.now();

    function update(timestamp) {
        const elapsed = timestamp - start;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 4);
        const current = Math.round(eased * target);
        el.textContent = current + (suffix || "");
        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

function initCounters() {
    const heroStats = document.querySelectorAll(".stat-number[data-target]");
    heroStats.forEach(el => {
        animateCounter(el, parseInt(el.dataset.target), "");
    });

    const statCards = document.querySelectorAll(".stat-value[data-target]");
    statCards.forEach(el => {
        const suffix = el.classList.contains("suffix-ms") ? "ms"
                     : el.classList.contains("suffix-pct") ? "%"
                     : "";
        animateCounter(el, parseInt(el.dataset.target), suffix);
    });
}

/* ── Intersection Observer for animations ────────── */
(function () {
    // Counter trigger on hero stats
    const heroSection = document.querySelector(".hero-stats");
    if (heroSection) {
        const obs = new IntersectionObserver(entries => {
            entries.forEach(e => {
                if (e.isIntersecting) {
                    initCounters();
                    obs.disconnect();
                }
            });
        }, { threshold: 0.5 });
        obs.observe(heroSection);
    }

    // Stats section counters
    const statsSection = document.querySelector(".stats-section");
    if (statsSection) {
        const obs2 = new IntersectionObserver(entries => {
            entries.forEach(e => {
                if (e.isIntersecting) {
                    document.querySelectorAll(".stat-value[data-target]").forEach(el => {
                        const suffix = el.classList.contains("suffix-ms") ? "ms"
                                     : el.classList.contains("suffix-pct") ? "%"
                                     : "";
                        animateCounter(el, parseInt(el.dataset.target), suffix);
                    });
                    obs2.disconnect();
                }
            });
        }, { threshold: 0.3 });
        obs2.observe(statsSection);
    }

    // Fade-in for feature cards
    const fadeEls = document.querySelectorAll("[data-aos]");
    const fadeObs = new IntersectionObserver(entries => {
        entries.forEach(e => {
            if (e.isIntersecting) {
                const delay = e.target.dataset.delay || 0;
                setTimeout(() => {
                    e.target.style.opacity = "1";
                    e.target.style.transform = "translateY(0)";
                }, parseInt(delay));
                fadeObs.unobserve(e.target);
            }
        });
    }, { threshold: 0.15 });

    fadeEls.forEach(el => {
        el.style.opacity = "0";
        el.style.transform = "translateY(20px)";
        el.style.transition = "opacity 0.6s ease, transform 0.6s ease";
        fadeObs.observe(el);
    });
})();

/* ── Smooth anchor scrolling ─────────────────────── */
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener("click", function (e) {
        const target = document.querySelector(this.getAttribute("href"));
        if (target) {
            e.preventDefault();
            target.scrollIntoView({ behavior: "smooth", block: "start" });
        }
    });
});
