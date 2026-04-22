# =============================================================================
# tests/unit/application/test_evaluer_demande_credit_use_case.py
# uv run pytest tests/unit/application/test_evaluer_demande_credit_use_case.py
#
# Tests unitaires du cas d'utilisation EvaluerDemandeCreditUseCase.
#
# Arquitectura Hexagonal — lo que se testea aquí:
#   - El UseCase orquesta correctamente (no contiene lógica técnica)
#   - Delega el scoring al port ICreditScorer
#   - Delega la journalisation al port IJournaliseurPredictions
#   - Gestiona errores sin bloquear la respuesta al cliente
#
# Estrategia de mocking:
#   - ICreditScorer      → MagicMock (simula ONNX/MLflow)
#   - IJournaliseurPredictions → MagicMock (simula escritura JSONL)
#   - DemandeCredit      → fixture con valores reales mínimos
#   - DecisionCredit     → MagicMock (retornado por el scoreur mock)
# =============================================================================

import pytest
from unittest.mock import MagicMock, patch, call

from src.api.domain.entities              import DemandeCredit, DecisionCredit
from src.api.ports.i_credit_scorer        import ICreditScorer
from src.api.ports.i_prediction_logger    import IJournaliseurPredictions
from src.api.application.evaluate_credit_use_case import EvaluerDemandeCreditUseCase


# =============================================================================
# Helpers
# =============================================================================

def _make_decision(score: float = 0.35, decision: str = "REFUS") -> MagicMock:
    """Crea un DecisionCredit mock con valores coherentes."""
    dec = MagicMock(spec=DecisionCredit)
    dec.score          = MagicMock()
    dec.score.valeur   = score
    dec.decision       = MagicMock()
    dec.decision.value = decision
    dec.latence_ms     = 12.5
    return dec


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_scoreur():
    """Scoreur ONNX simulado — siempre está listo y predice correctamente."""
    scoreur             = MagicMock(spec=ICreditScorer)
    scoreur.est_pret    = True
    scoreur.predire.return_value = _make_decision()
    return scoreur

@pytest.fixture
def mock_journaliseur():
    """Journaliseur JSONL simulado — escritura sin efectos secundarios."""
    return MagicMock(spec=IJournaliseurPredictions)

@pytest.fixture
def demande() -> DemandeCredit:
    return DemandeCredit(
        ext_source_1        = 0.5,
        ext_source_2        = 0.6,
        ext_source_3        = 0.4,
        paymnt_ratio_mean   = 0.1,
        age                 = 35,
        cc_drawings_mean    = 0.0,
        paymnt_delay_mean   = 2.0,
        pos_months_mean     = 12.0,
        goods_price         = 450000.0,
        education_type      = "Higher education",
        code_gender         = "F",
        bureau_credit_total = 5.0,
        max_dpd             = 0.0,
        amt_credit          = 500000.0,
        amt_annuity         = 25000.0,
        cc_balance_mean     = 0.0,
        years_employed      = 10,
        phone_change_days   = 365.0,
        region_rating       = 2,
        bureau_debt_mean    = 1000.0
        # No pongas 'type_pret' ni 'revenu' aquí porque DemandeCredit no los tiene.
    )

@pytest.fixture
def demande_V1() -> DemandeCredit:
    """Demande de crédit minimale válida para todos los tests."""
    return DemandeCredit(
        ext_source_1        = 0.5,
        ext_source_2        = 0.6,
        ext_source_3        = 0.4,
        type_pret           = "Cash loans",
        objet_pret          = "Unaccompanied",
        type_residence      = "House / apartment",
        code_gender         = "M",
        education_type      = "Secondary / secondary special",
        revenu              = 135000.0,
        montant_pret        = 450000.0,
        amt_annuity         = 22500.0,
        amt_goods_price     = 450000.0,
        paymnt_ratio_mean   = 0.95,
        paymnt_delay_mean   = 2.1,
        max_dpd             = 0.0,
        age                 = -12000,
        days_employed       = -3000,
        bureau_credit_total = 5,
        bureau_debt_mean    = 20000.0,
        pos_months_mean     = 18.0,
        cc_drawings_mean    = 3.5,
        cc_balance_mean     = 12000.0,
        phone_change_days   = -500,
        region_rating       = 2,
    )

@pytest.fixture
def use_case(mock_scoreur, mock_journaliseur) -> EvaluerDemandeCreditUseCase:
    return EvaluerDemandeCreditUseCase(
        scoreur      = mock_scoreur,
        journaliseur = mock_journaliseur,
    )


# =============================================================================
# Tests — Construcción e inyección de dependencias
# =============================================================================

class TestConstructeur:

    def test_use_case_instanciado_correctamente(self, mock_scoreur, mock_journaliseur):
        uc = EvaluerDemandeCreditUseCase(mock_scoreur, mock_journaliseur)
        assert uc is not None

    def test_scoreur_inyectado(self, use_case, mock_scoreur):
        assert use_case._scoreur is mock_scoreur

    def test_journaliseur_inyectado(self, use_case, mock_journaliseur):
        assert use_case._journaliseur is mock_journaliseur


# =============================================================================
# Tests — Flujo nominal (scoreur listo, predicción exitosa)
# =============================================================================

class TestExecuterFluxNominal:

    def test_retorna_decision_credit(self, use_case, demande):
        result = use_case.executer(demande)
        assert result is not None

    def test_retorna_el_resultado_del_scoreur(self, use_case, mock_scoreur, demande):
        decision_esperada = _make_decision(score=0.72, decision="REFUS")
        mock_scoreur.predire.return_value = decision_esperada

        result = use_case.executer(demande)

        assert result is decision_esperada

    def test_scoreur_predire_llamado_una_vez(self, use_case, mock_scoreur, demande):
        use_case.executer(demande)
        mock_scoreur.predire.assert_called_once()

    def test_scoreur_predire_llamado_con_la_demande(self, use_case, mock_scoreur, demande):
        use_case.executer(demande)
        mock_scoreur.predire.assert_called_once_with(demande)

    def test_journaliseur_llamado_una_vez(self, use_case, mock_journaliseur, demande):
        use_case.executer(demande)
        mock_journaliseur.journaliser.assert_called_once()

    def test_journaliseur_recibe_demande_y_decision(
            self, use_case, mock_scoreur, mock_journaliseur, demande
    ):
        decision = _make_decision()
        mock_scoreur.predire.return_value = decision

        use_case.executer(demande)

        mock_journaliseur.journaliser.assert_called_once_with(demande, decision)


# =============================================================================
# Tests — Scoreur no listo (est_pret = False)
# =============================================================================

class TestScoreurNonPret:

    def test_lanza_runtime_error_si_scoreur_no_listo(
            self, mock_scoreur, mock_journaliseur, demande
    ):
        mock_scoreur.est_pret = False
        uc = EvaluerDemandeCreditUseCase(mock_scoreur, mock_journaliseur)

        with pytest.raises(RuntimeError):
            uc.executer(demande)

    def test_mensaje_error_contiene_info_util(
            self, mock_scoreur, mock_journaliseur, demande
    ):
        mock_scoreur.est_pret = False
        uc = EvaluerDemandeCreditUseCase(mock_scoreur, mock_journaliseur)

        with pytest.raises(RuntimeError, match="scoreur"):
            uc.executer(demande)

    def test_predire_no_llamado_si_scoreur_no_listo(
            self, mock_scoreur, mock_journaliseur, demande
    ):
        mock_scoreur.est_pret = False
        uc = EvaluerDemandeCreditUseCase(mock_scoreur, mock_journaliseur)

        with pytest.raises(RuntimeError):
            uc.executer(demande)

        mock_scoreur.predire.assert_not_called()

    def test_journaliseur_no_llamado_si_scoreur_no_listo(
            self, mock_scoreur, mock_journaliseur, demande
    ):
        mock_scoreur.est_pret = False
        uc = EvaluerDemandeCreditUseCase(mock_scoreur, mock_journaliseur)

        with pytest.raises(RuntimeError):
            uc.executer(demande)

        mock_journaliseur.journaliser.assert_not_called()


# =============================================================================
# Tests — Journalisation échoue (IOError) → ne bloque pas la réponse
# =============================================================================

class TestJournalisationEchouee:

    def test_decision_retornada_aunque_journalisation_falla(
            self, use_case, mock_scoreur, mock_journaliseur, demande
    ):
        """El UseCase NO debe propagar el IOError del journaliseur."""
        decision = _make_decision()
        mock_scoreur.predire.return_value      = decision
        mock_journaliseur.journaliser.side_effect = IOError("Disco lleno")

        # No debe lanzar excepción
        result = use_case.executer(demande)

        assert result is decision

    def test_ioerror_journaliseur_no_propaga(
            self, use_case, mock_journaliseur, demande
    ):
        mock_journaliseur.journaliser.side_effect = IOError("JSONL no disponible")

        # Si propagara, el test fallaría — esperamos que no lo haga
        try:
            use_case.executer(demande)
        except IOError:
            pytest.fail("EvaluerDemandeCreditUseCase propagó IOError del journaliseur")

    def test_scoreur_predire_llamado_aunque_journalisation_falla(
            self, use_case, mock_scoreur, mock_journaliseur, demande
    ):
        mock_journaliseur.journaliser.side_effect = IOError("Error")

        use_case.executer(demande)

        mock_scoreur.predire.assert_called_once_with(demande)


# =============================================================================
# Tests — Valores de la decisión propagados correctamente
# =============================================================================

class TestValoresDecision:

    def test_score_alto_refus(self, use_case, mock_scoreur, demande):
        mock_scoreur.predire.return_value = _make_decision(score=0.85, decision="REFUS")
        result = use_case.executer(demande)
        assert result.score.valeur == 0.85
        assert result.decision.value == "REFUS"

    def test_score_bajo_accord(self, use_case, mock_scoreur, demande):
        mock_scoreur.predire.return_value = _make_decision(score=0.12, decision="ACCORD")
        result = use_case.executer(demande)
        assert result.score.valeur == 0.12
        assert result.decision.value == "ACCORD"

    def test_latence_propagada(self, use_case, mock_scoreur, demande):
        decision           = _make_decision()
        decision.latence_ms = 42.7
        mock_scoreur.predire.return_value = decision

        result = use_case.executer(demande)

        assert result.latence_ms == 42.7


# =============================================================================
# Tests — Llamadas múltiples (idempotencia del UseCase)
# =============================================================================

class TestLlamadasMultiples:

    def test_dos_ejecuciones_llaman_predire_dos_veces(
            self, use_case, mock_scoreur, demande
    ):
        use_case.executer(demande)
        use_case.executer(demande)
        assert mock_scoreur.predire.call_count == 2

    def test_dos_ejecuciones_llaman_journaliser_dos_veces(
            self, use_case, mock_journaliseur, demande
    ):
        use_case.executer(demande)
        use_case.executer(demande)
        assert mock_journaliseur.journaliser.call_count == 2

    def test_resultados_independientes_entre_llamadas(
            self, use_case, mock_scoreur, demande
    ):
        d1 = _make_decision(score=0.2, decision="ACCORD")
        d2 = _make_decision(score=0.9, decision="REFUS")
        mock_scoreur.predire.side_effect = [d1, d2]

        r1 = use_case.executer(demande)
        r2 = use_case.executer(demande)

        assert r1.score.valeur == 0.2
        assert r2.score.valeur == 0.9