const navToggle = document.querySelector(".nav-toggle");
const navLinks = document.querySelector(".nav-links");
const navbar = document.querySelector("[data-navbar]");

if (navToggle && navLinks) {
    navToggle.addEventListener("click", () => {
        const isOpen = navLinks.classList.toggle("is-open");
        navToggle.setAttribute("aria-expanded", String(isOpen));
    });

    navLinks.addEventListener("click", (event) => {
        if (event.target.matches("a")) {
            navLinks.classList.remove("is-open");
            navToggle.setAttribute("aria-expanded", "false");
        }
    });
}

window.addEventListener("scroll", () => {
    navbar?.classList.toggle("is-scrolled", window.scrollY > 8);
});

const revealObserver = new IntersectionObserver(
    (entries) => {
        entries.forEach((entry) => {
            if (entry.isIntersecting) {
                entry.target.classList.add("is-visible");
                revealObserver.unobserve(entry.target);
            }
        });
    },
    { threshold: 0.12 }
);

document.querySelectorAll(".reveal").forEach((element) => revealObserver.observe(element));

document.querySelectorAll("[data-toast]").forEach((toast) => {
    const close = toast.querySelector("[data-toast-close]");
    const dismiss = () => {
        toast.style.opacity = "0";
        toast.style.transform = "translateY(-6px)";
        window.setTimeout(() => toast.remove(), 180);
    };

    close?.addEventListener("click", dismiss);
    window.setTimeout(dismiss, 4600);
});

document.querySelectorAll("[data-prediction-form]").forEach((form) => {
    const cityInput = form.querySelector("[data-location-combobox]");
    const zipcodeInput = form.querySelector("[data-location-zipcode]");
    const latitudeInput = form.querySelector("[data-location-lat]");
    const longitudeInput = form.querySelector("[data-location-long]");
    const readout = form.querySelector("[data-location-readout]");
    const cityOptions = [...form.querySelectorAll("#city-options option")];

    function syncLocation() {
        if (!cityInput || !zipcodeInput || !latitudeInput || !longitudeInput) {
            return;
        }

        const selected = cityOptions.find(
            (option) => option.value.toLowerCase() === cityInput.value.trim().toLowerCase()
        );

        if (!selected) {
            return;
        }

        zipcodeInput.value = selected.dataset.zipcode || zipcodeInput.value;
        latitudeInput.value = selected.dataset.lat || latitudeInput.value;
        longitudeInput.value = selected.dataset.long || longitudeInput.value;

        if (readout) {
            readout.innerHTML = `
                <span>Market code</span>
                <strong>${zipcodeInput.value}</strong>
                <p>${latitudeInput.value}, ${longitudeInput.value}</p>
            `;
        }
    }

    cityInput?.addEventListener("input", syncLocation);
    cityInput?.addEventListener("change", syncLocation);
    syncLocation();
});

function animateCounter(element) {
    const target = Number(element.dataset.counter || 0);
    let current = 0;
    const steps = 32;
    const increment = target / steps;

    const tick = () => {
        current += increment;
        if (current >= target) {
            element.textContent = String(target);
            return;
        }
        element.textContent = String(Math.round(current));
        window.requestAnimationFrame(tick);
    };

    tick();
}

const counterObserver = new IntersectionObserver(
    (entries) => {
        entries.forEach((entry) => {
            if (entry.isIntersecting) {
                animateCounter(entry.target);
                counterObserver.unobserve(entry.target);
            }
        });
    },
    { threshold: 0.65 }
);

document.querySelectorAll("[data-counter]").forEach((counter) => counterObserver.observe(counter));

function validateField(input) {
    const wrapper = input.closest(".field");
    const message = wrapper?.querySelector("small");
    let error = "";

    if (input.validity.valueMissing) {
        error = "Required field";
    } else if (input.validity.badInput || input.validity.typeMismatch) {
        error = "Enter a valid value";
    } else if (input.type === "number" && Number(input.value) < 0 && input.name !== "long") {
        error = "Value cannot be negative";
    }

    wrapper?.classList.toggle("has-error", Boolean(error));
    if (message) {
        message.textContent = error;
    }
    return !error;
}

document.querySelectorAll("[data-prediction-form]").forEach((form) => {
    const inputs = [...form.querySelectorAll("input[required]")];

    inputs.forEach((input) => {
        input.addEventListener("blur", () => validateField(input));
        input.addEventListener("input", () => {
            if (input.closest(".field")?.classList.contains("has-error")) {
                validateField(input);
            }
        });
    });

    form.addEventListener("submit", (event) => {
        const isValid = inputs.every(validateField);
        if (!isValid) {
            event.preventDefault();
            form.querySelector(".has-error input")?.focus();
            return;
        }

        form.classList.add("is-loading");
        form.querySelector("button[type='submit']")?.setAttribute("disabled", "disabled");
    });
});

const searchInput = document.querySelector("[data-table-search]");
const recordsTable = document.querySelector("[data-records-table]");
const paginationStatus = document.querySelector("[data-pagination-status]");

if (searchInput && recordsTable) {
    const rows = [...recordsTable.querySelectorAll("tbody tr")];

    searchInput.addEventListener("input", () => {
        const query = searchInput.value.trim().toLowerCase();
        let visible = 0;

        rows.forEach((row) => {
            const match = row.textContent.toLowerCase().includes(query);
            row.hidden = !match;
            if (match) {
                visible += 1;
            }
        });

        if (paginationStatus) {
            paginationStatus.textContent = query ? `${visible} matching records` : "Showing all records";
        }
    });
}

const deleteModal = document.querySelector("[data-delete-modal]");
const deleteForm = document.querySelector("[data-delete-form]");

document.querySelectorAll("[data-delete-trigger]").forEach((button) => {
    button.addEventListener("click", () => {
        if (deleteForm) {
            deleteForm.action = button.dataset.deleteAction;
        }
        deleteModal?.classList.add("is-open");
        deleteModal?.setAttribute("aria-hidden", "false");
    });
});

document.querySelector("[data-modal-close]")?.addEventListener("click", () => {
    deleteModal?.classList.remove("is-open");
    deleteModal?.setAttribute("aria-hidden", "true");
});

deleteModal?.addEventListener("click", (event) => {
    if (event.target === deleteModal) {
        deleteModal.classList.remove("is-open");
        deleteModal.setAttribute("aria-hidden", "true");
    }
});
