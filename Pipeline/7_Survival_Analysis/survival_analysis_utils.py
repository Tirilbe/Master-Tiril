from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from statsmodels.stats.multitest import multipletests

try:
	from utils.network_and_gcc import AnalysisContext  # pyright: ignore[reportMissingImports]
except ModuleNotFoundError:
	import sys

	NETWORK_ANALYSIS_DIR = Path(__file__).resolve().parents[1] / "4_Network_Analysis"
	if str(NETWORK_ANALYSIS_DIR) not in sys.path:
		sys.path.insert(0, str(NETWORK_ANALYSIS_DIR))
	from utils.network_and_gcc import AnalysisContext  # pyright: ignore[reportMissingImports]


PLACEHOLDER_VALUES = {
	"",
	"'--",
	"--",
	"na",
	"n/a",
	"not reported",
	"not applicable",
	"unknown",
}

@dataclass(slots=True)
class SurvivalDatasetInputs:
	dataset: str
	expression: pd.DataFrame
	clinical: pd.DataFrame
	module_table: pd.DataFrame


def _clean_scalar(value: Any) -> Any:
	if pd.isna(value):
		return pd.NA
	if isinstance(value, str):
		cleaned = value.strip()
		if cleaned.lower() in PLACEHOLDER_VALUES:
			return pd.NA
		return cleaned
	return value


def _safe_float(value: Any) -> float:
	cleaned = _clean_scalar(value)
	if pd.isna(cleaned):
		return np.nan
	if isinstance(cleaned, (int, float, np.integer, np.floating)):
		return float(cleaned)
	text = str(cleaned).strip()
	match = re.search(r"-?\d+(?:\.\d+)?", text)
	if match is None:
		return np.nan
	return float(match.group(0))


def _stage_to_numeric(value: Any) -> float:
	cleaned = _clean_scalar(value)
	if pd.isna(cleaned):
		return np.nan
	text = str(cleaned).upper()
	if "STAGE IV" in text:
		return 4.0
	if "STAGE III" in text:
		return 3.0
	if "STAGE II" in text:
		return 2.0
	if "STAGE I" in text:
		return 1.0
	return np.nan


def _encode_gender(value: Any) -> float:
	cleaned = _clean_scalar(value)
	if pd.isna(cleaned):
		return np.nan
	text = str(cleaned).strip().lower()
	if text.startswith("m"):
		return 1.0
	if text.startswith("f"):
		return 0.0
	return np.nan


def _adjust_pvalues(values: pd.Series) -> pd.Series:
	numeric = pd.to_numeric(values, errors="coerce")
	adjusted = pd.Series(np.nan, index=values.index, dtype=float)
	valid_mask = numeric.notna()
	if valid_mask.any():
		adjusted.loc[valid_mask] = multipletests(numeric.loc[valid_mask], method="fdr_bh")[1]
	return adjusted


def _load_expression_with_symbols(path: Path) -> pd.DataFrame:
	matrix = pd.read_csv(path)
	gene_col = matrix.columns[0]
	matrix = matrix.rename(columns={gene_col: "gene_symbol"})
	matrix["gene_symbol"] = matrix["gene_symbol"].map(_clean_scalar)
	matrix = matrix.dropna(subset=["gene_symbol"])
	matrix["gene_symbol"] = matrix["gene_symbol"].astype(str)
	matrix = matrix.drop_duplicates(subset=["gene_symbol"], keep="first")
	matrix = matrix.set_index("gene_symbol")
	matrix = matrix.apply(pd.to_numeric, errors="coerce")
	return matrix


def _load_rna_expression(ctx: AnalysisContext) -> pd.DataFrame:
	symbol_path = ctx.notebook_output_path("06_single_gene_lookup", "RNA_cancer_expression_symbols.csv")
	if symbol_path.exists():
		matrix = _load_expression_with_symbols(symbol_path)
		if any(str(sample).startswith("C3") for sample in matrix.columns):
			return matrix

	fallback_path = ctx.project_root / "data" / "processed" / "rna_cancer_FINAL_FINAL.txt"
	if fallback_path.exists():
		matrix = pd.read_csv(fallback_path, sep=None, engine="python")
		gene_col = matrix.columns[0]
		matrix = matrix.rename(columns={gene_col: "gene_symbol"})
		matrix["gene_symbol"] = matrix["gene_symbol"].map(_clean_scalar)
		matrix = matrix.dropna(subset=["gene_symbol"])
		matrix["gene_symbol"] = matrix["gene_symbol"].astype(str)
		matrix = matrix.drop_duplicates(subset=["gene_symbol"], keep="first")
		matrix = matrix.set_index("gene_symbol")
		matrix = matrix.apply(pd.to_numeric, errors="coerce")
		return matrix

	raise FileNotFoundError(
		"No CPTAC RNA cancer expression matrix was found for the survival workflow."
	)


def _load_protein_expression(ctx: AnalysisContext) -> pd.DataFrame:
	final_final_path = ctx.project_root / "data" / "processed" / "proteome" / "proteome_cancer_FINAL_FINAL_with_header.txt"
	if final_final_path.exists():
		matrix = pd.read_csv(final_final_path, sep=None, engine="python")
		gene_col = matrix.columns[0]
		matrix = matrix.rename(columns={gene_col: "gene_symbol"})
		matrix["gene_symbol"] = matrix["gene_symbol"].map(_clean_scalar)
		matrix = matrix.dropna(subset=["gene_symbol"])
		matrix["gene_symbol"] = matrix["gene_symbol"].astype(str)
		matrix = matrix.drop_duplicates(subset=["gene_symbol"], keep="first")
		matrix = matrix.set_index("gene_symbol")
		matrix = matrix.apply(pd.to_numeric, errors="coerce")
		return matrix

	symbol_path = ctx.notebook_output_path("06_single_gene_lookup", "Protein_cancer_expression_symbols.csv")
	if symbol_path.exists():
		return _load_expression_with_symbols(symbol_path)

	raw_path = ctx.project_root / "data" / "processed" / "proteome" / "proteome_cancer_counts_FINAL_KNN_with_header.txt"
	matrix = pd.read_csv(raw_path, sep=None, engine="python")
	gene_col = matrix.columns[0]
	matrix = matrix.rename(columns={gene_col: "gene_symbol"}).set_index("gene_symbol")
	matrix = matrix.apply(pd.to_numeric, errors="coerce")
	return matrix


def _prepare_cptac_clinical(path: Path) -> pd.DataFrame:
	clinical = pd.read_csv(path, sep="\t", low_memory=False)
	clinical = clinical.rename(columns={
		"case_id": "sample_key",
		"age": "age",
		"sex": "gender",
		"Neoplastic_cellularity": "neoplastic_cellularity",
		"Stromal_fraction": "stromal_fraction",
		"Inflammation_fraction": "inflammation_fraction",
		"follow_up_days": "time",
		"tumor_stage_pathological": "stage",
	}).copy()
	clinical["age"] = clinical["age"].map(_safe_float)
	clinical["time"] = clinical["time"].map(_safe_float)
	clinical["neoplastic_cellularity"] = clinical["neoplastic_cellularity"].map(_safe_float)
	clinical["stromal_fraction"] = clinical["stromal_fraction"].map(_safe_float)
	clinical["inflammation_fraction"] = clinical["inflammation_fraction"].map(_safe_float)
	clinical["event"] = clinical["vital_status"].astype(str).str.lower().eq("deceased").astype(float)
	clinical["stage_numeric"] = clinical["stage"].map(_stage_to_numeric)
	clinical["gender_binary"] = clinical["gender"].map(_encode_gender)
	clinical = clinical.dropna(subset=["sample_key"])
	return clinical


def _load_dataset_inputs(ctx: AnalysisContext, dataset: str) -> SurvivalDatasetInputs:
	if dataset == "RNA":
		expression = _load_rna_expression(ctx)
		clinical = _prepare_cptac_clinical(ctx.clinical_files["cptac"])
		module_table = pd.read_csv(
			ctx.notebook_output_path("modules", "RNA_Leiden_Modules_1.0", "module_table.csv")
		)
	elif dataset == "PROTEIN":
		expression = _load_protein_expression(ctx)
		clinical = _prepare_cptac_clinical(ctx.clinical_files["cptac"])
		module_table = pd.read_csv(
			ctx.notebook_output_path("modules", "Protein_Leiden_Modules_1.0", "module_table.csv")
		)
	else:
		raise ValueError(f"Unsupported dataset '{dataset}'.")

	expression = expression.groupby(level=0).mean()
	common_samples = [sample for sample in expression.columns if sample in set(clinical["sample_key"])]
	expression = expression.loc[:, common_samples]
	clinical = clinical[clinical["sample_key"].isin(common_samples)].copy()
	return SurvivalDatasetInputs(
		dataset=dataset,
		expression=expression,
		clinical=clinical,
		module_table=module_table,
	)


def load_module_eigengenes(path: Path, condition: str = "Cancer") -> pd.DataFrame:
	eigengenes = pd.read_csv(path, index_col=0)
	eigengenes.index = eigengenes.index.map(_clean_scalar)
	eigengenes = eigengenes.loc[eigengenes.index.notna()].copy()

	if "condition" in eigengenes.columns:
		mask = eigengenes["condition"].astype(str).str.lower() == condition.lower()
		eigengenes = eigengenes.loc[mask].drop(columns=["condition"])

	eigengenes = eigengenes.groupby(level=0).mean(numeric_only=True)
	eigengenes.index.name = "sample_key"
	eigengenes.columns = [str(column) for column in eigengenes.columns]
	eigengenes = eigengenes.apply(pd.to_numeric, errors="coerce")
	return eigengenes


def build_survival_dataset_from_eigengenes(clinical: pd.DataFrame, eigengenes: pd.DataFrame) -> pd.DataFrame:
	base = clinical.loc[:, ["sample_key", "time", "event"]].copy()
	base = base.rename(columns={"sample_key": "sample", "time": "os_time", "event": "os_event"})
	base = base.dropna(subset=["sample", "os_time", "os_event"])
	base["sample"] = base["sample"].map(_clean_scalar)
	base = base.dropna(subset=["sample"])
	base = base.drop_duplicates(subset=["sample"], keep="first")

	survival = base.merge(eigengenes, left_on="sample", right_index=True, how="inner")
	survival = survival.sort_values("sample").reset_index(drop=True)
	return survival


def _module_columns_from_survival_matrix(survival_matrix: pd.DataFrame) -> list[str]:
	reserved = {"sample", "os_time", "os_event"}
	return [column for column in survival_matrix.columns if column not in reserved]


def build_kaplan_meier_groups(survival_matrix: pd.DataFrame, module: str | int) -> tuple[pd.DataFrame, float]:
	module_name = str(module)
	if module_name not in survival_matrix.columns:
		raise KeyError(f"Module '{module_name}' was not found in the survival matrix.")

	frame = survival_matrix.loc[:, ["sample", "os_time", "os_event", module_name]].copy()
	frame = frame.rename(columns={module_name: "module_eigengene"})
	frame["os_time"] = pd.to_numeric(frame["os_time"], errors="coerce")
	frame["os_event"] = pd.to_numeric(frame["os_event"], errors="coerce")
	frame["module_eigengene"] = pd.to_numeric(frame["module_eigengene"], errors="coerce")
	frame = frame.dropna(subset=["os_time", "os_event", "module_eigengene"]).copy()

	if frame.empty or frame["module_eigengene"].nunique(dropna=True) < 2:
		raise ValueError(f"Module '{module_name}' does not have enough variation for a Kaplan-Meier split.")

	threshold = float(frame["module_eigengene"].median())
	frame["group"] = np.where(frame["module_eigengene"] <= threshold, "Low ME", "High ME")
	return frame, threshold


def build_feature_kaplan_meier_groups(
	feature: pd.Series,
	clinical: pd.DataFrame,
	feature_name: str,
	low_label: str,
	high_label: str,
) -> tuple[pd.DataFrame, float]:
	frame = clinical.copy()
	frame = frame.merge(feature.rename("score"), left_on="sample_key", right_index=True, how="inner")
	frame = frame.dropna(subset=["score", "time", "event"]).copy()
	frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
	frame = frame.dropna(subset=["score"])

	if frame.empty or frame["score"].nunique(dropna=True) < 2:
		raise ValueError(f"Feature '{feature_name}' does not have enough variation for a Kaplan-Meier split.")

	threshold = float(frame["score"].median())
	frame["group"] = np.where(frame["score"] <= threshold, low_label, high_label)
	return frame, threshold


def _kaplan_meier_direction(low: pd.DataFrame, high: pd.DataFrame) -> str:
	low_median = low.loc[low["os_event"] == 1, "os_time"].median()
	high_median = high.loc[high["os_event"] == 1, "os_time"].median()

	if not np.isnan(low_median) and not np.isnan(high_median) and low_median != high_median:
		return "High ME worse" if high_median < low_median else "High ME better"

	low_event_rate = low["os_event"].mean()
	high_event_rate = high["os_event"].mean()
	if high_event_rate > low_event_rate:
		return "High ME worse"
	if high_event_rate < low_event_rate:
		return "High ME better"
	return "No clear direction"


def run_kaplan_meier_for_module(
	survival_matrix: pd.DataFrame,
	module: str | int,
	dataset: str,
) -> dict[str, Any]:
	module_name = str(module)
	frame, threshold = build_kaplan_meier_groups(survival_matrix, module_name)
	low = frame.loc[frame["group"] == "Low ME"].copy()
	high = frame.loc[frame["group"] == "High ME"].copy()

	result = {
		"dataset": dataset,
		"module": module_name,
		"n_patients": int(len(frame)),
		"n_events": int(frame["os_event"].sum()),
		"n_low": int(len(low)),
		"n_high": int(len(high)),
		"score_threshold": float(threshold),
		"km_p_value": np.nan,
		"direction": "No clear direction",
	}

	if low.empty or high.empty:
		return result

	try:
		km_test = logrank_test(
			low["os_time"],
			high["os_time"],
			event_observed_A=low["os_event"],
			event_observed_B=high["os_event"],
		)
		result["km_p_value"] = float(km_test.p_value)
	except Exception:
		pass

	result["direction"] = _kaplan_meier_direction(low, high)
	return result


def run_kaplan_meier_screen(survival_matrix: pd.DataFrame, dataset: str) -> pd.DataFrame:
	records: list[dict[str, Any]] = []
	for module_name in _module_columns_from_survival_matrix(survival_matrix):
		try:
			records.append(run_kaplan_meier_for_module(survival_matrix, module_name, dataset=dataset))
		except ValueError:
			continue

	results = pd.DataFrame(records)
	if results.empty:
		return results
	results["km_p_value_fdr"] = _adjust_pvalues(results["km_p_value"])
	return results.sort_values(["km_p_value_fdr", "km_p_value", "module"], na_position="last").reset_index(drop=True)


def plot_kaplan_meier_for_module(
	survival_matrix: pd.DataFrame,
	module: str | int,
	dataset: str,
	ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes, dict[str, Any]]:
	module_name = str(module)
	frame, _ = build_kaplan_meier_groups(survival_matrix, module_name)
	result = run_kaplan_meier_for_module(survival_matrix, module_name, dataset=dataset)

	if ax is None:
		figure, ax = plt.subplots(figsize=(6, 4))
	else:
		figure = ax.figure

	for group_name, color in (("Low ME", "#1f77b4"), ("High ME", "#d62728")):
		group_frame = frame.loc[frame["group"] == group_name]
		kmf = KaplanMeierFitter()
		kmf.fit(group_frame["os_time"], event_observed=group_frame["os_event"], label=group_name)
		kmf.plot_survival_function(ax=ax, ci_show=False, color=color)

	formatted_p = "nan" if pd.isna(result["km_p_value"]) else f"{result['km_p_value']:.3g}"
	ax.set_title(f"{dataset} Module {module_name}\np={formatted_p} | {result['direction']}")
	ax.set_xlabel("Overall survival time")
	ax.set_ylabel("Survival probability")
	ax.grid(alpha=0.2)
	return figure, ax, result


def save_kaplan_meier_plots(
	survival_matrix: pd.DataFrame,
	dataset: str,
	output_dir: Path,
	fdr_threshold: float = 0.1,
) -> tuple[pd.DataFrame, list[Path]]:
	results = run_kaplan_meier_screen(survival_matrix, dataset=dataset)
	plot_dir = output_dir / f"kaplan_meier_{dataset.lower()}"
	plot_dir.mkdir(parents=True, exist_ok=True)
	for existing_plot in plot_dir.glob("*.png"):
		existing_plot.unlink()

	significant_results = results.loc[results["km_p_value_fdr"].fillna(1.0) < fdr_threshold].copy()

	saved_paths: list[Path] = []
	for module in significant_results["module"]:
		figure, ax, result = plot_kaplan_meier_for_module(survival_matrix, module=module, dataset=dataset)
		module_name = str(result["module"])
		file_path = plot_dir / f"{dataset.lower()}_module_{module_name}_kaplan_meier.png"
		figure.savefig(file_path, dpi=300, bbox_inches="tight")
		plt.close(figure)
		saved_paths.append(file_path)

	return results, saved_paths


def plot_kaplan_meier_for_gene(
	gene: str,
	inputs: SurvivalDatasetInputs,
	gene_results: pd.DataFrame,
	ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes, dict[str, Any]]:
	gene_name = str(gene)
	if gene_name not in inputs.expression.index:
		raise KeyError(f"Gene '{gene_name}' was not found in the {inputs.dataset} expression matrix.")

	frame, _ = build_feature_kaplan_meier_groups(
		inputs.expression.loc[gene_name],
		inputs.clinical,
		feature_name=gene_name,
		low_label="Low expression",
		high_label="High expression",
	)

	if ax is None:
		figure, ax = plt.subplots(figsize=(6, 4))
	else:
		figure = ax.figure

	for group_name, color in (("Low expression", "#1f77b4"), ("High expression", "#d62728")):
		group_frame = frame.loc[frame["group"] == group_name]
		kmf = KaplanMeierFitter()
		kmf.fit(group_frame["time"], event_observed=group_frame["event"], label=group_name)
		kmf.plot_survival_function(ax=ax, ci_show=False, color=color)

	result_row = gene_results.loc[gene_results["gene"].astype(str) == gene_name]
	if result_row.empty:
		result = {
			"gene": gene_name,
			"km_p_value": np.nan,
			"km_p_value_fdr": np.nan,
			"direction": "No clear direction",
		}
	else:
		result = result_row.iloc[0].to_dict()

	formatted_p = "nan" if pd.isna(result.get("km_p_value")) else f"{result['km_p_value']:.3g}"
	ax.set_title(f"{inputs.dataset} {gene_name}\np={formatted_p} | {result.get('direction', 'No clear direction')}")
	ax.set_xlabel("Overall survival time")
	ax.set_ylabel("Survival probability")
	ax.grid(alpha=0.2)
	return figure, ax, result


def save_significant_gene_kaplan_meier_plots(
	inputs: SurvivalDatasetInputs,
	gene_results: pd.DataFrame,
	output_dir: Path,
	fdr_threshold: float = 0.1,
	plot_subdir: str | None = None,
) -> list[Path]:
	plot_dir = output_dir / (plot_subdir or f"kaplan_meier_{inputs.dataset.lower()}_genes")
	plot_dir.mkdir(parents=True, exist_ok=True)
	for existing_plot in plot_dir.glob("*.png"):
		existing_plot.unlink()

	significant_results = gene_results.loc[gene_results["km_p_value_fdr"].fillna(1.0) < fdr_threshold].copy()
	saved_paths: list[Path] = []
	for gene in significant_results["gene"].astype(str):
		figure, ax, result = plot_kaplan_meier_for_gene(gene, inputs=inputs, gene_results=gene_results)
		file_path = plot_dir / f"{inputs.dataset.lower()}_{gene}_kaplan_meier.png"
		figure.savefig(file_path, dpi=300, bbox_inches="tight")
		plt.close(figure)
		saved_paths.append(file_path)

	return saved_paths


def run_cox_proportional_hazards_for_module(
	survival_matrix: pd.DataFrame,
	module: str | int,
	dataset: str,
) -> dict[str, Any]:
	module_name = str(module)
	if module_name not in survival_matrix.columns:
		raise KeyError(f"Module '{module_name}' was not found in the survival matrix.")

	frame = survival_matrix.loc[:, ["os_time", "os_event", module_name]].copy()
	frame = frame.rename(columns={module_name: "module_eigengene"})
	frame["os_time"] = pd.to_numeric(frame["os_time"], errors="coerce")
	frame["os_event"] = pd.to_numeric(frame["os_event"], errors="coerce")
	frame["module_eigengene"] = pd.to_numeric(frame["module_eigengene"], errors="coerce")
	frame = frame.dropna(subset=["os_time", "os_event", "module_eigengene"]).copy()

	result = {
		"dataset": dataset,
		"module": module_name,
		"n_patients": int(len(frame)),
		"n_events": int(frame["os_event"].sum()),
		"cox_hr": np.nan,
		"cox_ci_lower": np.nan,
		"cox_ci_upper": np.nan,
		"cox_p_value": np.nan,
		"direction": "No clear direction",
	}

	if frame.empty or frame["module_eigengene"].nunique(dropna=True) < 2:
		return result

	try:
		cph = CoxPHFitter()
		cph.fit(frame, duration_col="os_time", event_col="os_event")
		summary = cph.summary.loc["module_eigengene"]
		result["cox_hr"] = float(summary["exp(coef)"])
		result["cox_ci_lower"] = float(summary["exp(coef) lower 95%"])
		result["cox_ci_upper"] = float(summary["exp(coef) upper 95%"])
		result["cox_p_value"] = float(summary["p"])

		hazard_ratio = result["cox_hr"]
		if hazard_ratio > 1.0:
			result["direction"] = "High ME worse"
		elif hazard_ratio < 1.0:
			result["direction"] = "High ME better"
	except Exception:
		pass

	return result


def run_cox_proportional_hazards_screen(survival_matrix: pd.DataFrame, dataset: str) -> pd.DataFrame:
	records: list[dict[str, Any]] = []
	for module_name in _module_columns_from_survival_matrix(survival_matrix):
		records.append(run_cox_proportional_hazards_for_module(survival_matrix, module_name, dataset=dataset))

	results = pd.DataFrame(records)
	if results.empty:
		return results
	results["cox_p_value_fdr"] = _adjust_pvalues(results["cox_p_value"])
	return results.sort_values(["cox_p_value_fdr", "cox_p_value", "module"], na_position="last").reset_index(drop=True)


def _top_modules_from_cox_results(cox_results: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
	if cox_results.empty:
		return cox_results.copy()
	ranking = cox_results.copy()
	ranking = ranking.sort_values(["cox_p_value_fdr", "cox_p_value", "module"], na_position="last").reset_index(drop=True)
	return ranking.head(top_n).copy()


def run_multivariable_cox_for_module(
	survival_matrix: pd.DataFrame,
	clinical: pd.DataFrame,
	module: str | int,
	dataset: str,
) -> dict[str, Any]:
	module_name = str(module)
	if module_name not in survival_matrix.columns:
		raise KeyError(f"Module '{module_name}' was not found in the survival matrix.")

	clinical_covariates = clinical.loc[:, [
		"sample_key",
		"age",
		"stage_numeric",
		"neoplastic_cellularity",
		"stromal_fraction",
		"inflammation_fraction",
	]].copy()
	frame = survival_matrix.loc[:, ["sample", "os_time", "os_event", module_name]].copy()
	frame = frame.rename(columns={module_name: "module_eigengene"})
	frame = frame.merge(clinical_covariates, left_on="sample", right_on="sample_key", how="left")
	frame = frame.drop(columns=["sample_key"])

	for column in [
		"os_time",
		"os_event",
		"module_eigengene",
		"age",
		"stage_numeric",
		"neoplastic_cellularity",
		"stromal_fraction",
		"inflammation_fraction",
	]:
		frame[column] = pd.to_numeric(frame[column], errors="coerce")
	frame = frame.dropna(subset=[
		"os_time",
		"os_event",
		"module_eigengene",
		"age",
		"stage_numeric",
		"neoplastic_cellularity",
		"stromal_fraction",
		"inflammation_fraction",
	]).copy()

	result = {
		"dataset": dataset,
		"module": module_name,
		"covariates": "age;stage_numeric;neoplastic_cellularity;stromal_fraction;inflammation_fraction",
		"n_patients": int(len(frame)),
		"n_events": int(frame["os_event"].sum()) if not frame.empty else 0,
		"multivariable_cox_hr": np.nan,
		"multivariable_cox_ci_lower": np.nan,
		"multivariable_cox_ci_upper": np.nan,
		"multivariable_cox_p_value": np.nan,
		"direction": "No clear direction",
	}

	if frame.empty or len(frame) < 20 or frame["module_eigengene"].nunique(dropna=True) < 2:
		return result

	for covariate in [
		"age",
		"stage_numeric",
		"neoplastic_cellularity",
		"stromal_fraction",
		"inflammation_fraction",
	]:
		if frame[covariate].nunique(dropna=True) <= 1:
			return result

	cox_frame = frame[[
		"os_time",
		"os_event",
		"module_eigengene",
		"age",
		"stage_numeric",
		"neoplastic_cellularity",
		"stromal_fraction",
		"inflammation_fraction",
	]].copy()
	try:
		cph = CoxPHFitter()
		cph.fit(cox_frame, duration_col="os_time", event_col="os_event")
		summary = cph.summary.loc["module_eigengene"]
		result["multivariable_cox_hr"] = float(summary["exp(coef)"])
		result["multivariable_cox_ci_lower"] = float(summary["exp(coef) lower 95%"])
		result["multivariable_cox_ci_upper"] = float(summary["exp(coef) upper 95%"])
		result["multivariable_cox_p_value"] = float(summary["p"])

		hazard_ratio = result["multivariable_cox_hr"]
		if hazard_ratio > 1.0:
			result["direction"] = "High ME worse"
		elif hazard_ratio < 1.0:
			result["direction"] = "High ME better"
	except Exception:
		pass

	return result


def run_top_multivariable_cox(
	survival_matrix: pd.DataFrame,
	clinical: pd.DataFrame,
	cox_results: pd.DataFrame,
	dataset: str,
	top_n: int = 5,
) -> pd.DataFrame:
	top_modules = _top_modules_from_cox_results(cox_results, top_n=top_n)
	if top_modules.empty:
		return pd.DataFrame()

	records: list[dict[str, Any]] = []
	for rank, (_, row) in enumerate(top_modules.iterrows(), start=1):
		module_name = str(row["module"])
		result = run_multivariable_cox_for_module(survival_matrix, clinical, module=module_name, dataset=dataset)
		records.append(
			{
				"rank": rank,
				"module": module_name,
				"dataset": dataset,
				"univariable_cox_hr": row.get("cox_hr", np.nan),
				"univariable_cox_p_value": row.get("cox_p_value", np.nan),
				"univariable_cox_p_value_fdr": row.get("cox_p_value_fdr", np.nan),
				"univariable_direction": row.get("direction", pd.NA),
				"covariates": result["covariates"],
				"multivariable_n_patients": result["n_patients"],
				"multivariable_n_events": result["n_events"],
				"multivariable_cox_hr": result["multivariable_cox_hr"],
				"multivariable_cox_ci_lower": result["multivariable_cox_ci_lower"],
				"multivariable_cox_ci_upper": result["multivariable_cox_ci_upper"],
				"multivariable_cox_p_value": result["multivariable_cox_p_value"],
				"multivariable_direction": result["direction"],
			}
		)

	results = pd.DataFrame(records)
	if results.empty:
		return results
	results["multivariable_cox_p_value_fdr"] = _adjust_pvalues(results["multivariable_cox_p_value"])
	return results.sort_values(["rank", "multivariable_cox_p_value"], na_position="last").reset_index(drop=True)


def compute_module_scores(module_table: pd.DataFrame, expression: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, int], dict[int, int]]:
	grouped = module_table.groupby("Module")["Gene"].apply(list)
	scores: dict[int, pd.Series] = {}
	n_genes_total: dict[int, int] = {}
	n_genes_used: dict[int, int] = {}

	available_genes = set(expression.index)
	for module_id, genes in grouped.items():
		cleaned_genes = pd.Series(genes).dropna().astype(str).tolist()
		module_genes = [gene for gene in cleaned_genes if gene in available_genes]
		n_genes_total[int(module_id)] = len(cleaned_genes)
		n_genes_used[int(module_id)] = len(module_genes)
		if not module_genes:
			continue
		scores[int(module_id)] = expression.loc[module_genes].mean(axis=0)

	score_frame = pd.DataFrame(scores)
	score_frame.index.name = "sample_key"
	return score_frame, n_genes_total, n_genes_used


def run_survival_test(
	feature: pd.Series,
	clinical: pd.DataFrame,
	feature_name: str,
	tier: str,
	dataset: str,
	module: int | None = None,
	n_genes_total: int | None = None,
	n_genes_used: int | None = None,
) -> dict[str, Any]:
	frame = clinical.copy()
	frame = frame.merge(feature.rename("score"), left_on="sample_key", right_index=True, how="inner")
	frame = frame.dropna(subset=["score", "time", "event"]).copy()
	frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
	frame = frame.dropna(subset=["score"])

	result = {
		"dataset": dataset,
		"module": module,
		"tier": tier,
		"feature": feature_name,
		"n_patients": int(len(frame)),
		"n_events": int(frame["event"].sum()),
		"n_low": np.nan,
		"n_high": np.nan,
		"score_threshold": np.nan,
		"km_p_value": np.nan,
		"cox_hr": np.nan,
		"cox_ci_lower": np.nan,
		"cox_ci_upper": np.nan,
		"cox_p_value": np.nan,
		"multivariable_n_patients": np.nan,
		"multivariable_n_events": np.nan,
		"multivariable_cox_hr": np.nan,
		"multivariable_cox_ci_lower": np.nan,
		"multivariable_cox_ci_upper": np.nan,
		"multivariable_cox_p_value": np.nan,
		"n_genes_total": n_genes_total,
		"n_genes_used": n_genes_used,
	}

	if frame.empty or frame["score"].nunique(dropna=True) < 2:
		return result

	threshold = float(frame["score"].median())
	frame["group"] = np.where(frame["score"] <= threshold, "low", "high")
	low = frame[frame["group"] == "low"]
	high = frame[frame["group"] == "high"]
	result["n_low"] = int(len(low))
	result["n_high"] = int(len(high))
	result["score_threshold"] = threshold

	if len(low) > 0 and len(high) > 0:
		try:
			km_test = logrank_test(
				low["time"],
				high["time"],
				event_observed_A=low["event"],
				event_observed_B=high["event"],
			)
			result["km_p_value"] = float(km_test.p_value)
		except Exception:
			pass

	cox_frame = frame[["time", "event", "score"]].copy()
	try:
		cph = CoxPHFitter()
		cph.fit(cox_frame, duration_col="time", event_col="event")
		summary = cph.summary.loc["score"]
		result["cox_hr"] = float(summary["exp(coef)"])
		result["cox_ci_lower"] = float(summary["exp(coef) lower 95%"])
		result["cox_ci_upper"] = float(summary["exp(coef) upper 95%"])
		result["cox_p_value"] = float(summary["p"])
	except Exception:
		pass

	covariates = ["age", "stage_numeric", "gender_binary"]
	available_covariates = [
		covariate
		for covariate in covariates
		if covariate in frame.columns and frame[covariate].notna().sum() >= max(10, len(frame) // 2)
	]
	if available_covariates:
		multivariable = frame[["time", "event", "score", *available_covariates]].dropna().copy()
		valid_covariates = [
			covariate for covariate in available_covariates if multivariable[covariate].nunique(dropna=True) > 1
		]
		multivariable = multivariable[["time", "event", "score", *valid_covariates]]
		if len(multivariable) >= 20 and multivariable["score"].nunique(dropna=True) > 1:
			try:
				cph = CoxPHFitter()
				cph.fit(multivariable, duration_col="time", event_col="event")
				summary = cph.summary.loc["score"]
				result["multivariable_n_patients"] = int(len(multivariable))
				result["multivariable_n_events"] = int(multivariable["event"].sum())
				result["multivariable_cox_hr"] = float(summary["exp(coef)"])
				result["multivariable_cox_ci_lower"] = float(summary["exp(coef) lower 95%"])
				result["multivariable_cox_ci_upper"] = float(summary["exp(coef) upper 95%"])
				result["multivariable_cox_p_value"] = float(summary["p"])
			except Exception:
				pass

	return result


def run_module_survival_screen(inputs: SurvivalDatasetInputs, tier: str = "tier_1_module") -> pd.DataFrame:
	score_frame, n_genes_total, n_genes_used = compute_module_scores(inputs.module_table, inputs.expression)
	records = []
	for module_id in sorted(score_frame.columns):
		if n_genes_used.get(int(module_id), 0) == 0:
			continue
		records.append(
			run_survival_test(
				score_frame[module_id],
				inputs.clinical,
				feature_name=f"Module {module_id}",
				tier=tier,
				dataset=inputs.dataset,
				module=int(module_id),
				n_genes_total=n_genes_total.get(int(module_id)),
				n_genes_used=n_genes_used.get(int(module_id)),
			)
		)

	results = pd.DataFrame(records)
	if results.empty:
		return results
	results["direction"] = "No clear direction"
	results.loc[results["cox_hr"] > 1.0, "direction"] = "High expression worse"
	results.loc[results["cox_hr"] < 1.0, "direction"] = "High expression better"
	results["km_p_value_fdr"] = _adjust_pvalues(results["km_p_value"])
	results = results.sort_values(["km_p_value_fdr", "km_p_value", "cox_p_value"], na_position="last").reset_index(drop=True)
	return results


def significant_hits(results: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
	if results.empty:
		return results.copy()
	km_mask = results["km_p_value_fdr"].fillna(1.0) <= alpha
	cox_mask = (
		results["multivariable_cox_p_value"].fillna(results["cox_p_value"]).fillna(1.0) <= alpha
	)
	return results.loc[km_mask | cox_mask].copy()


def run_cross_omics_module_validation(
	source_module_table: pd.DataFrame,
	source_km_results: pd.DataFrame,
	source_cox_results: pd.DataFrame,
	target_inputs: SurvivalDatasetInputs,
	source_dataset: str,
	target_dataset: str,
	fdr_threshold: float = 0.1,
	tier: str = "tier_2_cross_omics_module",
) -> pd.DataFrame:
	if source_km_results.empty or source_cox_results.empty:
		return pd.DataFrame()

	source_summary = source_km_results.loc[:, ["module", "km_p_value", "km_p_value_fdr", "direction"]].rename(
		columns={
			"km_p_value": "source_km_p_value",
			"km_p_value_fdr": "source_km_p_value_fdr",
			"direction": "source_km_direction",
		}
	)
	source_summary = source_summary.merge(
		source_cox_results.loc[:, ["module", "cox_hr", "cox_p_value", "cox_p_value_fdr", "direction"]].rename(
			columns={
				"cox_hr": "source_cox_hr",
				"cox_p_value": "source_cox_p_value",
				"cox_p_value_fdr": "source_cox_p_value_fdr",
				"direction": "source_direction",
			}
		),
		on="module",
		how="outer",
	)
	source_summary["module"] = pd.to_numeric(source_summary["module"], errors="coerce")
	source_summary = source_summary.dropna(subset=["module"]).copy()
	source_summary["module"] = source_summary["module"].astype(int)

	significant_source = source_summary.loc[
		(source_summary["source_km_p_value_fdr"].fillna(1.0) <= fdr_threshold)
		| (source_summary["source_cox_p_value_fdr"].fillna(1.0) <= fdr_threshold)
	].copy()
	if significant_source.empty:
		return significant_source

	module_subset = source_module_table.loc[
		source_module_table["Module"].isin(significant_source["module"]),
		["Module", "Gene"],
	].copy()
	target_score_frame, n_genes_total, n_genes_used = compute_module_scores(module_subset, target_inputs.expression)

	records: list[dict[str, Any]] = []
	for module_id in significant_source["module"].tolist():
		module_key = int(module_id)
		feature = target_score_frame[module_key] if module_key in target_score_frame.columns else pd.Series(dtype=float)
		result = run_survival_test(
			feature,
			target_inputs.clinical,
			feature_name=f"{source_dataset} module {module_key} gene set",
			tier=tier,
			dataset=f"{target_dataset}_from_{source_dataset}",
			module=module_key,
			n_genes_total=n_genes_total.get(module_key),
			n_genes_used=n_genes_used.get(module_key),
		)
		records.append(result)

	target_results = pd.DataFrame(records)
	target_results["target_direction"] = "No clear direction"
	target_results.loc[target_results["cox_hr"] > 1.0, "target_direction"] = "High ME worse"
	target_results.loc[target_results["cox_hr"] < 1.0, "target_direction"] = "High ME better"
	target_results["target_km_p_value_fdr"] = _adjust_pvalues(target_results["km_p_value"])
	target_results["target_cox_p_value_fdr"] = _adjust_pvalues(target_results["cox_p_value"])
	target_results["target_multivariable_cox_p_value_fdr"] = _adjust_pvalues(target_results["multivariable_cox_p_value"])
	target_results = target_results.rename(
		columns={
			"dataset": "target_dataset_label",
			"tier": "target_tier",
			"feature": "target_feature",
			"n_patients": "target_n_patients",
			"n_events": "target_n_events",
			"n_low": "target_n_low",
			"n_high": "target_n_high",
			"score_threshold": "target_score_threshold",
			"km_p_value": "target_km_p_value",
			"cox_hr": "target_cox_hr",
			"cox_ci_lower": "target_cox_ci_lower",
			"cox_ci_upper": "target_cox_ci_upper",
			"cox_p_value": "target_cox_p_value",
			"multivariable_n_patients": "target_multivariable_n_patients",
			"multivariable_n_events": "target_multivariable_n_events",
			"multivariable_cox_hr": "target_multivariable_cox_hr",
			"multivariable_cox_ci_lower": "target_multivariable_cox_ci_lower",
			"multivariable_cox_ci_upper": "target_multivariable_cox_ci_upper",
			"multivariable_cox_p_value": "target_multivariable_cox_p_value",
			"n_genes_total": "target_n_genes_total",
			"n_genes_used": "target_n_genes_used",
		}
	)

	validated = significant_source.merge(target_results, on="module", how="left")
	validated["supported_in_target"] = (
		(validated["target_km_p_value_fdr"].fillna(1.0) <= fdr_threshold)
		| (validated["target_cox_p_value_fdr"].fillna(1.0) <= fdr_threshold)
		| (validated["target_multivariable_cox_p_value_fdr"].fillna(1.0) <= fdr_threshold)
	)
	validated["direction_concordant"] = validated["source_direction"].eq(validated["target_direction"])
	validated["cross_omics_supported"] = validated["supported_in_target"] & validated["direction_concordant"]
	validated["source_dataset"] = source_dataset
	validated["target_dataset"] = target_dataset
	validated = validated.sort_values(
		["cross_omics_supported", "supported_in_target", "target_cox_p_value_fdr", "target_km_p_value_fdr", "module"],
		ascending=[False, False, True, True, True],
		na_position="last",
	).reset_index(drop=True)
	return validated


def _module_gene_sets(module_table: pd.DataFrame) -> dict[int, set[str]]:
	grouped = module_table.groupby("Module")["Gene"].apply(list)
	return {
		int(module_id): set(pd.Series(genes).dropna().astype(str).tolist())
		for module_id, genes in grouped.items()
	}


def build_module_overlap_pairs(
	rna_module_table: pd.DataFrame,
	protein_module_table: pd.DataFrame,
	rna_results: pd.DataFrame,
	protein_results: pd.DataFrame,
	alpha: float = 0.05,
) -> pd.DataFrame:
	rna_modules = set(significant_hits(rna_results, alpha)["module"].dropna().astype(int))
	protein_modules = set(significant_hits(protein_results, alpha)["module"].dropna().astype(int))
	rna_gene_sets = _module_gene_sets(rna_module_table)
	protein_gene_sets = _module_gene_sets(protein_module_table)

	rows: list[dict[str, Any]] = []
	for rna_module in sorted(rna_modules):
		for protein_module in sorted(protein_modules):
			rna_genes = rna_gene_sets.get(int(rna_module), set())
			protein_genes = protein_gene_sets.get(int(protein_module), set())
			intersection = sorted(rna_genes & protein_genes)
			if not intersection:
				continue
			union_size = len(rna_genes | protein_genes)
			rows.append(
				{
					"RNA_module": int(rna_module),
					"Protein_module": int(protein_module),
					"jaccard": len(intersection) / union_size if union_size else 0.0,
					"intersection_size": len(intersection),
					"union_size": union_size,
					"validated_by_cox": True,
					"overlap_genes": ";".join(intersection),
				}
			)



def run_overlap_gene_screen(
	panel: pd.DataFrame,
	inputs: SurvivalDatasetInputs,
	tier: str = "tier_3_gene",
) -> pd.DataFrame:
	records = []
	for gene in panel["gene"].astype(str):
		if gene not in inputs.expression.index:
			continue
		result = run_survival_test(
			inputs.expression.loc[gene],
			inputs.clinical,
			feature_name=gene,
			tier=tier,
			dataset=inputs.dataset,
		)
		result["gene"] = gene
		records.append(result)

	results = pd.DataFrame(records)
	if results.empty:
		return results
	results["direction"] = "No clear direction"
	results.loc[results["cox_hr"] > 1.0, "direction"] = "High expression worse"
	results.loc[results["cox_hr"] < 1.0, "direction"] = "High expression better"
	results["km_p_value_fdr"] = _adjust_pvalues(results["km_p_value"])
	results = results.sort_values(["km_p_value_fdr", "km_p_value", "cox_p_value"], na_position="last").reset_index(drop=True)
	return results


def build_named_gene_panel(
	genes: list[str] | tuple[str, ...] | pd.Series,
	panel_label: str,
	candidate_group: str,
) -> pd.DataFrame:
	cleaned_genes: list[str] = []
	for gene in genes:
		cleaned = _clean_scalar(gene)
		if pd.isna(cleaned):
			continue
		gene_name = str(cleaned).strip()
		if not gene_name:
			continue
		cleaned_genes.append(gene_name)

	panel = pd.DataFrame({"gene": pd.Index(cleaned_genes, dtype=str).drop_duplicates().tolist()})
	if panel.empty:
		return panel
	panel["panel_label"] = panel_label
	panel["candidate_group"] = candidate_group
	return panel


def build_user_defined_gene_panel(genes: list[str] | tuple[str, ...] | pd.Series) -> pd.DataFrame:
	return build_named_gene_panel(genes, panel_label="User defined", candidate_group="user_defined")


def build_top_hub_gene_panel(
	ctx: AnalysisContext,
	top_n: int | None = 10,
	degree_threshold: int | float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	rna_degree_path = ctx.notebook_output_path("topology_and_hubs", "RNA_GCC_node_degrees.csv")
	protein_degree_path = ctx.notebook_output_path("topology_and_hubs", "Protein_GCC_node_degrees.csv")

	rna_top_hubs = pd.read_csv(rna_degree_path).rename(columns={"Node": "gene", "Degree": "rna_node_degree"})
	protein_top_hubs = pd.read_csv(protein_degree_path).rename(columns={"Node": "gene", "Degree": "protein_node_degree"})

	if degree_threshold is not None:
		rna_top_hubs = rna_top_hubs.loc[pd.to_numeric(rna_top_hubs["rna_node_degree"], errors="coerce") >= float(degree_threshold)]
		protein_top_hubs = protein_top_hubs.loc[
			pd.to_numeric(protein_top_hubs["protein_node_degree"], errors="coerce") >= float(degree_threshold)
		]
		panel_label = f"Top hub genes with >= {degree_threshold:g} edges"
	else:
		selected_top_n = 10 if top_n is None else int(top_n)
		rna_top_hubs = rna_top_hubs.head(selected_top_n)
		protein_top_hubs = protein_top_hubs.head(selected_top_n)
		panel_label = f"Top {selected_top_n} hub genes"

	ordered_genes = pd.Index(
		[
			*pd.Index(rna_top_hubs["gene"].astype(str)),
			*pd.Index(protein_top_hubs["gene"].astype(str)),
		],
		dtype=str,
	).drop_duplicates().tolist()
	panel = build_named_gene_panel(ordered_genes, panel_label=panel_label, candidate_group="top_hub")
	return panel, rna_top_hubs.reset_index(drop=True), protein_top_hubs.reset_index(drop=True)


def _match_shared_gene_panel(
	requested_genes: pd.DataFrame,
	rna_expression: pd.DataFrame,
	protein_expression: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
	rna_lookup = {str(gene).upper(): str(gene) for gene in rna_expression.index.astype(str)}
	protein_lookup = {str(gene).upper(): str(gene) for gene in protein_expression.index.astype(str)}

	shared_rows: list[dict[str, Any]] = []
	missing_genes: list[str] = []
	for _, requested_row in requested_genes.iterrows():
		gene = str(requested_row["gene"])
		key = gene.upper()
		if key in rna_lookup and key in protein_lookup:
			shared_rows.append(
				{
					"gene": rna_lookup[key],
					"panel_label": requested_row.get("panel_label", "Gene panel"),
					"candidate_group": requested_row.get("candidate_group", "gene_panel"),
				}
			)
		else:
			missing_genes.append(gene)

	shared_panel = pd.DataFrame(shared_rows).drop_duplicates(subset=["gene"], keep="first")
	return shared_panel.reset_index(drop=True), missing_genes


def annotate_user_defined_gene_panel(
	ctx: AnalysisContext,
	shared_panel: pd.DataFrame,
) -> pd.DataFrame:
	annotated_panel = shared_panel.copy()
	if annotated_panel.empty:
		annotated_panel["rna_node_degree"] = pd.Series(dtype="Int64")
		annotated_panel["protein_node_degree"] = pd.Series(dtype="Int64")
		annotated_panel["rna_module"] = pd.Series(dtype="Int64")
		annotated_panel["protein_module"] = pd.Series(dtype="Int64")
		return annotated_panel

	rna_inputs = _load_dataset_inputs(ctx, "RNA")
	protein_inputs = _load_dataset_inputs(ctx, "PROTEIN")

	rna_degree_path = ctx.notebook_output_path("topology_and_hubs", "RNA_GCC_node_degrees.csv")
	protein_degree_path = ctx.notebook_output_path("topology_and_hubs", "Protein_GCC_node_degrees.csv")

	rna_degrees = pd.read_csv(rna_degree_path).rename(columns={"Node": "gene", "Degree": "rna_node_degree"})
	protein_degrees = pd.read_csv(protein_degree_path).rename(columns={"Node": "gene", "Degree": "protein_node_degree"})

	rna_modules = rna_inputs.module_table.loc[:, ["Gene", "Module"]].rename(
		columns={"Gene": "gene", "Module": "rna_module"}
	)
	protein_modules = protein_inputs.module_table.loc[:, ["Gene", "Module"]].rename(
		columns={"Gene": "gene", "Module": "protein_module"}
	)

	annotated_panel = annotated_panel.merge(rna_degrees, on="gene", how="left")
	annotated_panel = annotated_panel.merge(protein_degrees, on="gene", how="left")
	annotated_panel = annotated_panel.merge(rna_modules, on="gene", how="left")
	annotated_panel = annotated_panel.merge(protein_modules, on="gene", how="left")

	for column in ("rna_node_degree", "protein_node_degree", "rna_module", "protein_module"):
		annotated_panel[column] = pd.to_numeric(annotated_panel[column], errors="coerce").astype("Int64")

	return annotated_panel


def run_named_gene_survival_panel(
	ctx: AnalysisContext,
	genes: list[str] | tuple[str, ...] | pd.Series,
	panel_label: str,
	candidate_group: str,
	tier: str = "tier_4_user_gene",
) -> dict[str, pd.DataFrame | list[str]]:
	requested_panel = build_named_gene_panel(genes, panel_label=panel_label, candidate_group=candidate_group)
	rna_inputs = _load_dataset_inputs(ctx, "RNA")
	protein_inputs = _load_dataset_inputs(ctx, "PROTEIN")
	shared_panel, missing_genes = _match_shared_gene_panel(
		requested_panel,
		rna_inputs.expression,
		protein_inputs.expression,
	)
	annotated_shared_panel = annotate_user_defined_gene_panel(ctx, shared_panel)

	rna_results = run_overlap_gene_screen(shared_panel, rna_inputs, tier=tier)
	protein_results = run_overlap_gene_screen(shared_panel, protein_inputs, tier=tier)

	return {
		"requested_gene_panel": requested_panel,
		"shared_gene_panel": annotated_shared_panel,
		"missing_genes": missing_genes,
		"rna_results": rna_results,
		"protein_results": protein_results,
	}


def run_user_defined_gene_survival_panel(
	ctx: AnalysisContext,
	genes: list[str] | tuple[str, ...] | pd.Series,
) -> dict[str, pd.DataFrame | list[str]]:
	return run_named_gene_survival_panel(
		ctx,
		genes,
		panel_label="User defined",
		candidate_group="user_defined",
		tier="tier_4_user_gene",
	)


def build_significant_module_km_panel(
	survival_matrix: pd.DataFrame,
	km_results: pd.DataFrame,
	dataset: str,
	output_dir: Path,
	fdr_threshold: float = 0.05,
	expected_modules: int = 3,
	save_figures: bool = True,
) -> dict[str, Any]:
	significant_modules = (
		km_results.loc[km_results["km_p_value_fdr"] < fdr_threshold]
		.sort_values(["km_p_value_fdr", "km_p_value", "module"])
		.reset_index(drop=True)
	)
	if len(significant_modules) != expected_modules:
		raise ValueError(
			f"Expected {expected_modules} significant {dataset} KM modules at FDR < {fdr_threshold}, "
			f"but found {len(significant_modules)}."
		)

	figure, axes = plt.subplots(1, expected_modules, figsize=(7.2, 2.85), sharex=True, sharey=True)
	for ax, (_, result_row) in zip(axes, significant_modules.iterrows()):
		module = result_row["module"]
		plot_kaplan_meier_for_module(
			survival_matrix,
			module=module,
			dataset=dataset,
			ax=ax,
		)
		ax.set_title(f"M{module}", fontsize=10, fontweight="bold", pad=7)
		ax.text(
			0.97,
			0.95,
			f"adj. p = {result_row['km_p_value_fdr']:.3g}",
			transform=ax.transAxes,
			ha="right",
			va="top",
			fontsize=7,
			fontweight="normal",
			bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2.0},
		)
		ax.set_xlabel("")
		ax.set_ylabel("")
		ax.set_xticks([0, 500, 1000])
		ax.tick_params(axis="both", labelsize=7)
		ax.grid(alpha=0.22, linewidth=0.7)
		for spine in ax.spines.values():
			spine.set_linewidth(0.9)

	handles, labels = axes[0].get_legend_handles_labels()
	for ax in axes:
		legend = ax.get_legend()
		if legend is not None:
			legend.remove()
	axes[0].set_ylabel("Survival probability", fontsize=8, fontweight="semibold", labelpad=3)
	if len(axes) > 1:
		axes[1].set_xlabel("Overall survival time (days)", fontsize=8, fontweight="semibold", labelpad=3)
	else:
		axes[0].set_xlabel("Overall survival time (days)", fontsize=8, fontweight="semibold", labelpad=3)
	figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False, fontsize=8)
	figure.tight_layout(rect=(0, 0, 1, 0.86), w_pad=0.55)

	output_prefix = dataset.lower()
	png_path = output_dir / f"{output_prefix}_significant_module_kaplan_meier_panel.png"
	pdf_path = output_dir / f"{output_prefix}_significant_module_kaplan_meier_panel.pdf"
	data_path = output_dir / f"{output_prefix}_significant_module_kaplan_meier_panel_data.csv"
	significant_modules.to_csv(data_path, index=False)
	if save_figures:
		figure.savefig(png_path, dpi=300, bbox_inches="tight")
		figure.savefig(pdf_path, bbox_inches="tight")

	return {
		"results": significant_modules,
		"figure": figure,
		"png_path": png_path,
		"pdf_path": pdf_path,
		"data_path": data_path,
	}


def _save_gene_panel_survival_outputs(
	ctx: AnalysisContext,
	output_stem: str,
	rna_results: pd.DataFrame,
	protein_results: pd.DataFrame,
	requested_panel: pd.DataFrame,
	shared_panel: pd.DataFrame,
	missing_genes: pd.DataFrame,
	fdr_threshold: float,
	rna_inputs: SurvivalDatasetInputs,
	protein_inputs: SurvivalDatasetInputs,
	save_plots: bool = True,
) -> dict[str, Any]:
	requested_panel.to_csv(ctx.output_dir / f"{output_stem}_requested_gene_panel.csv", index=False)
	shared_panel.to_csv(ctx.output_dir / f"{output_stem}_shared_gene_panel.csv", index=False)
	missing_genes.to_csv(ctx.output_dir / f"{output_stem}_missing_genes.csv", index=False)
	rna_results.to_csv(ctx.output_dir / f"rna_{output_stem}_gene_survival_results.csv", index=False)
	protein_results.to_csv(ctx.output_dir / f"protein_{output_stem}_gene_survival_results.csv", index=False)

	if save_plots:
		rna_plot_paths = save_significant_gene_kaplan_meier_plots(
			rna_inputs,
			rna_results,
			ctx.output_dir,
			fdr_threshold=fdr_threshold,
			plot_subdir=f"kaplan_meier_rna_{output_stem}_genes",
		)
		protein_plot_paths = save_significant_gene_kaplan_meier_plots(
			protein_inputs,
			protein_results,
			ctx.output_dir,
			fdr_threshold=fdr_threshold,
			plot_subdir=f"kaplan_meier_protein_{output_stem}_genes",
		)
	else:
		rna_plot_paths = []
		protein_plot_paths = []

	return {
		"requested_gene_panel": requested_panel,
		"shared_gene_panel": shared_panel,
		"missing_genes": missing_genes,
		"rna_results": rna_results,
		"protein_results": protein_results,
		"rna_plot_paths": rna_plot_paths,
		"protein_plot_paths": protein_plot_paths,
	}


def run_dataset_specific_gene_panel_pair(
	ctx: AnalysisContext,
	rna_genes: list[str] | tuple[str, ...] | pd.Series,
	protein_genes: list[str] | tuple[str, ...] | pd.Series,
	output_stem: str,
	rna_panel_label: str,
	protein_panel_label: str,
	rna_candidate_group: str,
	protein_candidate_group: str,
	tier: str,
	fdr_threshold: float = 0.1,
	save_plots: bool = True,
) -> dict[str, Any]:
	rna_inputs = _load_dataset_inputs(ctx, "RNA")
	protein_inputs = _load_dataset_inputs(ctx, "PROTEIN")

	rna_panel = build_named_gene_panel(rna_genes, panel_label=rna_panel_label, candidate_group=rna_candidate_group)
	protein_panel = build_named_gene_panel(
		protein_genes,
		panel_label=protein_panel_label,
		candidate_group=protein_candidate_group,
	)
	requested_panel = pd.concat([rna_panel, protein_panel], ignore_index=True)
	annotated_panel = annotate_user_defined_gene_panel(ctx, requested_panel)
	missing_rows = [
		{"dataset": "RNA", "gene": gene}
		for gene in rna_panel.get("gene", pd.Series(dtype=str)).astype(str)
		if gene not in rna_inputs.expression.index
	] + [
		{"dataset": "PROTEIN", "gene": gene}
		for gene in protein_panel.get("gene", pd.Series(dtype=str)).astype(str)
		if gene not in protein_inputs.expression.index
	]
	missing_genes = pd.DataFrame(missing_rows, columns=["dataset", "gene"])

	rna_results = run_overlap_gene_screen(rna_panel, rna_inputs, tier=tier)
	protein_results = run_overlap_gene_screen(protein_panel, protein_inputs, tier=tier)
	outputs = _save_gene_panel_survival_outputs(
		ctx,
		output_stem=output_stem,
		rna_results=rna_results,
		protein_results=protein_results,
		requested_panel=requested_panel,
		shared_panel=annotated_panel,
		missing_genes=missing_genes,
		fdr_threshold=fdr_threshold,
		rna_inputs=rna_inputs,
		protein_inputs=protein_inputs,
		save_plots=save_plots,
	)
	outputs["genes"] = requested_panel.get("gene", pd.Series(dtype=str)).astype(str).tolist()
	return outputs


def run_shared_gene_panel_survival(
	ctx: AnalysisContext,
	genes: list[str] | tuple[str, ...] | pd.Series,
	output_stem: str,
	panel_label: str,
	candidate_group: str,
	tier: str,
	fdr_threshold: float = 0.1,
	save_plots: bool = True,
) -> dict[str, Any]:
	requested_panel = build_named_gene_panel(genes, panel_label=panel_label, candidate_group=candidate_group)
	outputs = run_named_gene_survival_panel(
		ctx,
		requested_panel.get("gene", pd.Series(dtype=str)).astype(str).tolist(),
		panel_label=panel_label,
		candidate_group=candidate_group,
		tier=tier,
	)
	rna_inputs = _load_dataset_inputs(ctx, "RNA")
	protein_inputs = _load_dataset_inputs(ctx, "PROTEIN")
	missing_genes = pd.DataFrame({"gene": outputs["missing_genes"]})
	saved_outputs = _save_gene_panel_survival_outputs(
		ctx,
		output_stem=output_stem,
		rna_results=outputs["rna_results"],
		protein_results=outputs["protein_results"],
		requested_panel=outputs["requested_gene_panel"],
		shared_panel=outputs["shared_gene_panel"],
		missing_genes=missing_genes,
		fdr_threshold=fdr_threshold,
		rna_inputs=rna_inputs,
		protein_inputs=protein_inputs,
		save_plots=save_plots,
	)
	saved_outputs["genes"] = requested_panel.get("gene", pd.Series(dtype=str)).astype(str).tolist()
	return saved_outputs


def _load_overlap_panel_genes(
	ctx: AnalysisContext,
	notebook_name: str,
	file_name: str,
	agreement_label: str | None = None,
	gene_column: str = "Gene",
	agreement_column: str = "CSD_Agreement",
) -> list[str]:
	nodes_path = ctx.notebook_output_path(notebook_name, file_name)
	nodes = pd.read_csv(nodes_path)
	if agreement_label and agreement_column in nodes.columns:
		agreement = nodes[agreement_column].astype(str).str.strip().str.upper()
		nodes = nodes.loc[agreement.eq(agreement_label.upper())].copy()
	gene_series = nodes[gene_column].dropna().astype(str).str.strip()
	return gene_series.loc[gene_series.ne("")].drop_duplicates().tolist()


def run_overlap_file_gene_panel_survival(
	ctx: AnalysisContext,
	notebook_name: str,
	file_name: str,
	output_stem: str,
	panel_label: str,
	candidate_group: str,
	tier: str,
	agreement_label: str | None = None,
	fdr_threshold: float = 0.1,
	save_plots: bool = True,
) -> dict[str, Any]:
	genes = _load_overlap_panel_genes(
		ctx,
		notebook_name=notebook_name,
		file_name=file_name,
		agreement_label=agreement_label,
	)
	outputs = run_shared_gene_panel_survival(
		ctx,
		genes,
		output_stem=output_stem,
		panel_label=panel_label,
		candidate_group=candidate_group,
		tier=tier,
		fdr_threshold=fdr_threshold,
		save_plots=save_plots,
	)
	outputs["agreement_label"] = agreement_label or panel_label
	outputs["genes"] = genes
	return outputs


