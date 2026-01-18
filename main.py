from src.data_loading import load_student_data
from src.data_cleaning import clean_full_dataset
from src.feature_engineering import engineer_features
from src.modeling import build_interaction_regression_model, summarize_model
from src.analysis.model_analysis import compute_correlations, extract_model_effects
from src.visualization_basic import plot_distributions
from src.analysis.eda_advanced import apply_pca, apply_kmeans, plot_pca_clusters
from src.analysis.eda_components import eda_score_components
from src.visualization_advanced import plot_interaction_effect
from src.utils.logger import get_logger

#comments
logger = get_logger(__name__)


def main():
    logger.info("Starting stress-sleep-performance analysis pipeline")

    df_raw = load_student_data()

    df_clean = clean_full_dataset(df_raw)

    df_features = engineer_features(df_clean)

    plot_distributions(df_clean)

    eda_score_components(df_clean)

    correlations = compute_correlations(df_features)
    logger.info("Correlations: %s", correlations)

    pca_df = apply_pca(df_features)
    df_clustered = apply_kmeans(pca_df)
    plot_pca_clusters(pca_df, df_clustered["Cluster"])

    model = build_interaction_regression_model(df_features)

    effects = extract_model_effects(model)
    logger.info("Model effects: %s", effects)

    logger.info("Model summary:\n%s", summarize_model(model))

    plot_interaction_effect(df_features, model)

    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()