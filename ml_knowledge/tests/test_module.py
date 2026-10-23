import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
import unittest

from ml_knowledge.build import build, chunks
from ml_knowledge.context import (SelectionContext, audit_request, evaluate_atoms, pack, prepare_request)
from ml_knowledge.core import canonical, digest, file_hash, write_json
from ml_knowledge.memory import KnowledgeMemory, DEPENDENCIES
from ml_knowledge.normalize import Resolver
from ml_knowledge.runtime import ReferenceArgs, ReferenceStore


def document(identifier, text, *, library="scikit-learn", version="1.5.2", symbol=None, kind="documentation", **extra):
    text = text + "\n" if not text.endswith("\n") else text
    return {"id": identifier, "text": text, "doc_sha256": digest(text), "type": kind, "title": symbol or identifier,
            "library": library, "version": version, "symbols": [symbol] if symbol else [],
            "source_url": "https://example.test/exact/" + identifier, "source_kind": "test_fixture", "license": "test",
            "conditions": [], "warnings": [], "content_status": "complete", "applicability": "unknown", **extra}


def fixture(root, records=None):
    frozen = root / "frozen"; frozen.mkdir(parents=True)
    records = records or [
        document("long", "begin\n\n" + ("ordinary words " * 1200) + "\n\nUNIQUE_TAIL_MARKER\n\n" + "x" * 5000, symbol="sklearn.Long"),
        document("cat", "CatBoostClassifier.fit supports cat_features.\n\nNever fit on validation.", library="catboost", version="1.2.8", symbol="catboost.CatBoostClassifier.fit"),
        document("enc1", "Sklearn encoder cross fitting.", symbol="sklearn.preprocessing.TargetEncoder"),
        document("enc2", "Category encoder target encoding.", library="category-encoders", version="2.6.4", symbol="category_encoders.TargetEncoder"),
        document("mlp", "MLPClassifier.fit(self, X, y)\n\nsample_weight is not a parameter in this version.", symbol="sklearn.neural_network.MLPClassifier.fit", warnings=["Do not pass sample_weight to MLPClassifier.fit in 1.5.2."], aliases=["api_sklearn_neural_network_MLPClassifier_fit"]),
        document("method:target_encoding", "Target encoding with random cross-fitting.", kind="method", library="project", version="2", conditions=["Only independent rows"], warnings=["Random cross-fitting is not time-aware."], payload={"hypothesis":"Try target encoding", "references":["enc1"], "applicability_rules":[{"field":"validation","op":"not_in","value":["temporal"],"reason":"Random folds violate time ordering"}]})]
    with (frozen / "documents.jsonl").open("w") as stream:
        for record in sorted(records, key=lambda r:r["id"]):
            stream.write(canonical(record) + "\n")
    write_json(frozen / "expected.json", {"items":[]})
    write_json(frozen / "coverage.json", {"complete": False, "test_fixture": True})
    write_json(frozen / "inputs.lock.json", {"documents_sha256":file_hash(frozen/"documents.jsonl"), "expected_sha256":file_hash(frozen/"expected.json"), "coverage_sha256":file_hash(frozen/"coverage.json"), "requirements_sha256":digest("fixture")})
    destination = root / "corpus"; build(frozen, destination)
    return destination


class ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.corpus = fixture(cls.root)

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def setUp(self): self.store = ReferenceStore(self.corpus)
    def tearDown(self): self.store.close()

    def test_tail_is_indexed_in_fts_and_fallback(self):
        for enabled in (True,False):
            self.store.use_fts = enabled
            hits = self.store.search(ReferenceArgs(query="UNIQUE_TAIL_MARKER"))
            self.assertEqual(hits[0]["doc"],"long")
            self.assertGreater(hits[0]["start"],10000)

    def test_filter_precedes_limit(self):
        for enabled in (True,False):
            self.store.use_fts=enabled
            hits=self.store.search(ReferenceArgs(query="fit",library="catboost",version="1.2.8"))
            self.assertTrue(hits)
            self.assertEqual({x["doc"] for x in hits},{"cat"})

    def test_wrong_version(self):
        self.assertEqual(self.store.query(query="fit",library="catboost",version="999")["status"],"missing_version")

    def test_negative_all_modes(self):
        for mode in ("search","methods","recipes","search_space","api"):
            self.assertEqual(self.store.query(mode=mode,query="qzzplkrandomabsent")["status"],"not_found")

    def test_ambiguous_symbol(self):
        self.assertEqual(self.store.query(mode="api",query="TargetEncoder")["status"],"ambiguous_symbol")
        self.assertNotEqual(self.store.query(mode="api",query="TargetEncoder",library="category-encoders")["status"],"ambiguous_symbol")

    def test_budget_includes_every_character(self):
        for budget in (100,200,700,1500,3000,6000):
            response=self.store.query(mode="read",reference_id="long",max_chars=budget)
            self.assertLessEqual(len(canonical(response)),budget)
        response=self.store.query(mode="read",reference_id="long",max_chars=100)
        self.assertEqual(response["status"],"budget_too_small")
        self.assertGreater(response["min_chars"],100)

    def test_continuation_reconstructs_unicode_document(self):
        cursor=0;parts=[]
        while True:
            response=self.store.query(mode="read",reference_id="long",cursor=cursor,max_chars=2200)
            self.assertIn("entry",response)
            parts.append(response["entry"]["content"])
            cursor=response["next_cursor"]
            if not cursor:break
        self.assertEqual("".join(parts),self.store.document("long")["text"])

    def test_cursor_all_bindings(self):
        response=self.store.query(mode="read",reference_id="long",max_chars=2000,context={"validation":"random"})
        token=response["next_cursor"]
        self.assertIsInstance(token,str)
        for change in ({"query":"changed"},{"reference_id":"cat"},{"library":"catboost"},{"mode":"api"}):
            args={"mode":"read","reference_id":"long","cursor":token};args.update(change)
            self.assertEqual(self.store.query(args,context={"validation":"random"})["status"],"invalid_cursor")
        self.assertEqual(self.store.query(mode="read",reference_id="long",cursor=token,context={"validation":"time"})["status"],"stale_cursor")
        old=self.store.version;self.store.version="new-corpus"
        self.assertEqual(self.store.query(mode="read",reference_id="long",cursor=token,context={"validation":"random"})["status"],"stale_cursor")
        self.store.version=old

    def test_cursor_size_limit_can_change(self):
        token=self.store.query(mode="read",reference_id="long",max_chars=2000)["next_cursor"]
        self.assertIn("entry",self.store.query(mode="read",reference_id="long",cursor=token,max_chars=3500,limit=1))

    def test_numeric_and_tampered_cursors_rejected(self):
        for cursor in (1,True,-1,"bad-token",{},[] ):
            self.assertEqual(self.store.query(mode="read",reference_id="long",cursor=cursor)["status"],"invalid_cursor")

    def test_unknown_applicability_stays_unknown(self):
        result=self.store.query(mode="read",reference_id="mlp")
        self.assertTrue(result["answer_usable"])
        self.assertEqual(result["entry"]["applicability"],"unknown")

    def test_versioned_alias_is_preserved(self):
        self.assertEqual(self.store.query(mode="read",reference_id="api_sklearn_neural_network_MLPClassifier_fit")["entry"]["id"],"mlp")

    def test_incompatible_method(self):
        result=self.store.query(mode="methods",reference_id="target_encoding",context={"validation":"temporal"})
        self.assertEqual(result["entry"]["applicability"],"inapplicable")
        self.assertTrue(result["entry"]["applicability_reasons"])

    def test_passing_guard_does_not_verify_prose_prerequisites(self):
        result = self.store.query(mode="methods", reference_id="target_encoding", context={"validation": "random"})
        self.assertEqual(result["entry"]["applicability"], "unknown")

    def test_search_internal_fields_not_exposed(self):
        result=canonical(self.store.query(query="CatBoost",library="catboost"))
        self.assertNotIn('search_text',result)
        self.assertNotIn('postings',result)

    def test_corrupt_corpus_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            copy=Path(tmp)/"copy";shutil.copytree(self.corpus,copy)
            with (copy/"knowledge.sqlite").open("ab") as stream:stream.write(b"corrupt")
            with ReferenceStore(copy) as store:
                self.assertEqual(store.query(query="fit")["status"],"integrity_error")

    def test_corrupt_or_missing_lock_has_integrity_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            for content in ('not-json','{}'):
                (path/'knowledge.lock.json').write_text(content)
                with ReferenceStore(path) as store:self.assertEqual(store.query(query='fit')['status'],'integrity_error')
            (path/'knowledge.lock.json').unlink()
            with ReferenceStore(path) as store:self.assertEqual(store.query(query='fit')['status'],'integrity_error')

    def test_reproducible_clean_builds(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/"clean";build(self.root/"frozen",out)
            for path in self.corpus.iterdir():
                self.assertEqual(file_hash(path),file_hash(out/path.name))

    def test_lossless_chunks(self):
        text="Кириллица 🧪\n\n"+"long warning "*600+"\n\nend"
        self.assertEqual("".join(text[a:b] for a,b in chunks(text)),text)

    def test_filter_before_limit_with_many_other_library_hits(self):
        with tempfile.TemporaryDirectory() as tmp:
            docs=[document('other'+str(i),'fit token '*10,symbol='sklearn.Other'+str(i)) for i in range(80)]
            docs.append(document('wanted','fit',library='catboost',version='1.2.8',symbol='catboost.Target.fit'))
            with ReferenceStore(fixture(Path(tmp),docs)) as store:
                for fts in (True,False):
                    store.use_fts=fts
                    self.assertEqual(store.search(ReferenceArgs(query='fit',library='catboost',version='1.2.8'))[0]['doc'],'wanted')

    def test_known_warning_never_lost_as_usable(self):
        with tempfile.TemporaryDirectory() as tmp:
            text='Intro recommendation.\n\n'+('context '*500)+'\n\nWARNING: only when independent rows; never for temporal folds.'
            start=text.index('WARNING')
            doc=document('warning',text,required_spans=[{'start':start,'end':len(text),'kind':'warning_with_condition'}])
            with ReferenceStore(fixture(Path(tmp),[doc])) as store:
                response=store.query(mode='read',reference_id='warning',max_chars=1800)
                self.assertFalse(response['answer_usable'])
                self.assertTrue(response['entry']['required_reading'])

    def test_unicode_complete_response_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            with ReferenceStore(fixture(Path(tmp),[document('unicode','Кириллица 🧪 '*300)])) as store:
                cursor=0;parts=[]
                while True:
                    reply=store.query(mode='read',reference_id='unicode',cursor=cursor,max_chars=1700)
                    self.assertLessEqual(len(canonical(reply)),1700)
                    parts.append(reply['entry']['content']);cursor=reply['next_cursor']
                    if not cursor:break
                self.assertEqual(''.join(parts),store.document('unicode')['text'])

    def test_search_pagination_is_bound_and_changes_limit(self):
        first=self.store.query(query='encoder',limit=1,max_chars=6000)
        self.assertTrue(first['next_cursor'])
        second=self.store.query(query='encoder',limit=5,max_chars=6000,cursor=first['next_cursor'])
        self.assertNotEqual(first['entries'][0]['id'],second['entries'][0]['id'])
        self.assertEqual(self.store.query(query='different',cursor=first['next_cursor'])['status'],'invalid_cursor')


class MemoryContextTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.store=ReferenceStore(fixture(self.root));self.path=self.root/"memory.sqlite"
        self.memory=KnowledgeMemory(self.path)
        self.context={key:"fixed" for key in DEPENDENCIES};self.context["validation"]="random"

    def tearDown(self):
        self.memory.close();self.store.close();self.tmp.cleanup()

    def test_restart_keeps_substantive_constraint(self):
        doc=self.store.document("mlp");identifier=self.memory.fact(self.store,"mlp",0,len(doc["text"]),contraindications=doc["warnings"])
        n=self.memory.progress_count
        self.memory.provided(identifier,packet_sha256="test");self.memory.referenced(identifier,operation_id="other_tool")
        self.assertEqual(n,self.memory.progress_count)
        self.memory.close();self.memory=KnowledgeMemory(self.path)
        packet=pack([],memory=self.memory.active(self.store,self.context))
        request,audit=prepare_request([{"role":"user","content":"Continue after restart"}],packet,model="fixture",compressor=lambda r:r)
        self.assertTrue(audit["send_allowed"])
        self.assertIn("Do not pass sample_weight",request["messages"][-1]["content"])
        self.assertLessEqual(len(canonical(packet)),6000)

    def test_compressor_loss_is_blocked(self):
        result=self.store.query(mode="read",reference_id="mlp")
        packet=pack([result["entry"]])
        with self.assertRaises(ValueError):
            prepare_request([],packet,model="fixture",compressor=lambda r:{**r,"messages":[]})

    def test_fact_not_invalidated_by_pipeline_only(self):
        self.memory.fact(self.store,"mlp",0,30)
        self.context["pipeline_hash"]="different"
        active=self.memory.active(self.store,self.context)
        self.assertEqual(active[0]["verification"],"source_verified")

    def test_memory_excerpt_keeps_missing_warning_visible_after_restart(self):
        text = "Recommendation.\n\nWARNING: only independent rows; never temporal folds."
        start = text.index("WARNING")
        with tempfile.TemporaryDirectory() as tmp:
            doc = document("warning", text, required_spans=[{"start": start, "end": len(text), "kind": "warning_with_condition"}])
            with ReferenceStore(fixture(Path(tmp), [doc])) as store:
                self.memory.fact(store, "warning", 0, len("Recommendation."))
                self.memory.close(); self.memory = KnowledgeMemory(self.path)
                active = self.memory.active(store, self.context)
                self.assertEqual(active[0]["verification"], "source_verified")
                packet = pack([], memory=active)
                self.assertFalse(packet["answer_usable"])
                self.assertEqual(packet["materials"][0]["required_reading"][0]["start"], start)

    def test_memory_full_quote_of_partial_source_is_not_usable(self):
        with tempfile.TemporaryDirectory() as tmp:
            doc = document("partial", "Available source text", content_status="partial")
            with ReferenceStore(fixture(Path(tmp), [doc])) as store:
                self.memory.fact(store, "partial", 0, len(doc["text"]))
                packet = pack([], memory=self.memory.active(store, self.context))
                self.assertFalse(packet["answer_usable"])
                self.assertIn("unresolved", packet["materials"][0]["required_reading"][0]["reason"])

    def test_fact_automatically_retains_known_source_warnings(self):
        doc = self.store.document("mlp")
        self.memory.fact(self.store, "mlp", 0, len(doc["text"]))
        packet = pack([], memory=self.memory.active(self.store, self.context))
        self.assertIn(doc["warnings"][0], packet["materials"][0]["warnings"])
        self.assertIn("not automatically verified", packet["materials"][0]["verification_scope"])

    def test_possible_contradiction_is_visible_in_active_packet(self):
        doc = self.store.document("mlp")
        self.memory.fact(self.store, "mlp", 0, 30, topic="same-topic")
        self.memory.fact(self.store, "mlp", 30, len(doc["text"]), topic="same-topic")
        packet = pack([], memory=self.memory.active(self.store, self.context))
        self.assertFalse(packet["answer_usable"])
        self.assertTrue(all(material["conflicts"] for material in packet["materials"]))

    def test_packet_does_not_present_recommendation_without_required_source(self):
        method = self.store.query(mode="read", reference_id="method:target_encoding")["entry"]
        source = self.store.query(mode="read", reference_id="enc1")["entry"]
        method["required_material_ids"] = [source["id"]]
        source["content"] = "evidence " * 2000
        memory = [{"id": f"memory:{index}", "type": "experiment", "claim": "retained note",
            "source": "controller", "conditions": [], "contraindications": [], "verification": "source_verified",
            "applicability": "applicable", "answer_usable": True} for index in (1, 2)]
        packet = pack([method, source], memory=memory, priority_count=1)
        self.assertEqual(packet["status"], "partial")
        self.assertFalse(packet["answer_usable"])
        self.assertEqual(packet["required_reading"][0]["id"], source["id"])

    def test_required_source_precedes_optional_memory(self):
        method = self.store.query(mode="read", reference_id="method:target_encoding")["entry"]
        source = self.store.query(mode="read", reference_id="enc1")["entry"]
        method["required_material_ids"] = [source["id"]]
        memory = [{"id": "memory:" + str(index), "type": "experiment", "claim": "retained note",
            "source": "controller", "conditions": [], "contraindications": [], "verification": "source_verified",
            "applicability": "applicable", "answer_usable": True} for index in (1, 2)]
        packet = pack([method, source], memory=memory, priority_count=1)
        self.assertEqual([entry["id"] for entry in packet["materials"]], [method["id"], source["id"], "memory:1"])
        self.assertEqual(packet["status"], "available")
        self.assertTrue(packet["answer_usable"])

    def test_unrouted_method_keeps_required_source_ahead_of_memory(self):
        method = self.store.query(mode="read", reference_id="method:target_encoding")["entry"]
        source = self.store.query(mode="read", reference_id="enc1")["entry"]
        method["required_material_ids"] = [source["id"]]
        memory = [{"id": "memory:" + str(index), "type": "experiment", "claim": "retained note",
            "source": "controller", "conditions": [], "contraindications": [], "verification": "source_verified",
            "applicability": "applicable", "answer_usable": True} for index in (1, 2)]
        packet = pack([method, source], memory=memory, priority_count=0)
        self.assertEqual([entry["id"] for entry in packet["materials"]], [method["id"], source["id"], "memory:1"])
        self.assertEqual(packet["status"], "available")
        self.assertTrue(packet["answer_usable"])

    def test_validation_change_stales_recommendation(self):
        self.memory.recommendation(self.store,"target_encoding",context=self.context)
        changed={**self.context,"validation":"temporal","validation_hash":"new"}
        result=self.memory.current(self.store,changed)[0]
        self.assertEqual(result["applicability"],"stale_context")
        self.assertTrue(result["applicability_reasons"])
        self.assertEqual(self.memory.active(self.store,changed),[])

    def test_changed_support_stales_unchanged_recommendation(self):
        self.memory.recommendation(self.store, "target_encoding", context=self.context)
        method = self.store.document("method:target_encoding")
        with tempfile.TemporaryDirectory() as tmp:
            changed_source = document("enc1", "The source has changed", symbol="sklearn.preprocessing.TargetEncoder")
            with ReferenceStore(fixture(Path(tmp), [method, changed_source])) as store:
                record = self.memory.current(store, self.context)[0]
                self.assertEqual(record["verification"], "stale_source")
                self.assertIn("supporting_source_changed:enc1", record["stale_reasons"])
                self.assertEqual(self.memory.active(store, self.context), [])

    def test_hypothesis_never_becomes_fact(self):
        self.memory.hypothesis("Trees might help",observation="nonlinear fit pattern",expected_effect="better inner validation",test="one trial",context=self.context)
        record=self.memory.current(self.store,self.context)[0]
        self.assertEqual(record["type"],"agent_hypothesis")
        self.assertEqual(record["verification"],"unchecked")

    def test_timeout_is_not_quality_evidence(self):
        self.memory.experiment("e1",controller_lookup=lambda x:{"experiment_id":x,"status":"timeout"},context=self.context)
        self.assertIn("does not establish",self.memory.current(self.store,self.context)[0]["contraindications"][0])

    def test_application_requires_controller_evidence(self):
        with self.assertRaises(ValueError):
            self.memory.outcome("id",decision="applied",evidence={"reference_id":"mlp"},controller_verify=lambda d,e:False)

    def test_dependency_contract(self):
        with self.assertRaises(ValueError):
            self.memory.hypothesis("guess",observation="obs",expected_effect="effect",test="trial",context={})

    def test_oversized_critical_material_not_shortened(self):
        entry=self.store.query(mode="read",reference_id="mlp")["entry"]
        entry["warnings"]=["critical warning "*500]
        packet=pack([entry])
        self.assertFalse(packet["answer_usable"])
        self.assertTrue(packet["required_reading"])

    def test_sufficiency_and_or_after_transport(self):
        entry=self.store.query(mode="read",reference_id="mlp")["entry"];packet=pack([entry]);request,_=prepare_request([],packet,model="fixture")
        annotation={"versions":["1.5.2"],"atoms":[{"id":"signature","critical":True,"alternatives":[{"required_text":["MLPClassifier.fit(self, X, y)"]},{"required_text":["alternative documented signature"]}]},{"id":"warning","alternatives":[{"required_text":["Do not pass sample_weight"]}]}]}
        self.assertTrue(evaluate_atoms(request,annotation)["sufficient"])
        bad={"messages":[{"content":"MLPClassifier.fit(self, X, y) 1.5.2"}]}
        self.assertFalse(evaluate_atoms(bad,annotation)["sufficient"])

    def test_hidden_labels_not_accepted_by_selector(self):
        with self.assertRaises(ValueError):
            SelectionContext("classification",{"hidden_labels":[1,2]},"random","mlp",{})

    def test_project_card_cannot_become_documentation_fact(self):
        with self.assertRaises(ValueError):self.memory.fact(self.store,'method:target_encoding',0,20)

    def test_conflicting_hypotheses_preserved_in_history(self):
        for claim in ['Expected gain from encoding','Expected loss from encoding']:
            self.memory.hypothesis(claim,observation='fit study',expected_effect='measure',test='one trial',context=self.context,topic='encoding')
        self.assertEqual(len(self.memory.current(self.store,self.context)),2)
        self.assertTrue(any(e['kind']=='possible_contradiction' for e in self.memory.history()))

    def test_evaluator_binds_version_to_source(self):
        entry=self.store.query(mode='read',reference_id='mlp')['entry']
        annotation={'versions':['1.5.2'],'atoms':[{'id':'bound','critical':True,'alternatives':[{'library':'scikit-learn','version':'1.5.2','doc_sha256':entry['doc_sha256'],'required_text':['MLPClassifier.fit']}]}]}
        entry['version']='999';entry['content']+=' unrelated 1.5.2';packet=pack([entry]);request,_=prepare_request([],packet,model='test')
        self.assertFalse(evaluate_atoms(request,annotation)['sufficient'])


class NormalizationTests(unittest.TestCase):
    def test_include_resolves_and_reports_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/"part.txt").write_text("warning\nfull condition\n")
            (root/"doc.rst").write_text(".. include:: part.txt\n\n.. include:: absent.txt\n")
            resolver=Resolver(root,{})
            result=resolver.read("doc.rst")
            self.assertIn("full condition",result);self.assertTrue(resolver.gaps)

    def test_literalinclude_line_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/"sample.py").write_text("one\ntwo\nthree\nfour\n")
            (root/"doc.rst").write_text(".. literalinclude:: sample.py\n   :lines: 2-3\n")
            result=Resolver(root,{}).read("doc.rst")
            self.assertIn("two\nthree",result);self.assertNotIn("four",result)

    def test_include_traversal_not_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/"doc.rst").write_text(".. include:: ../../etc/passwd\n")
            resolver=Resolver(root,{})
            self.assertIn("UNRESOLVED",resolver.read("doc.rst"));self.assertTrue(resolver.gaps)


if __name__=="__main__":unittest.main()
