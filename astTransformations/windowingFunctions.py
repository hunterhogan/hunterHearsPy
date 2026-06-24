"""Generate PyTorch tensor windowing functions from NumPy windowing functions.

This module programmatically creates PyTorch tensor versions of windowing functions by transforming
existing functions from the `windowingFunctions` module. It generates functions that return PyTorch
tensors instead of NumPy arrays.
"""
from __future__ import annotations

from astToolkit import Be, Make, NodeTourist, parseLogicalPath2astModule, Then
from astToolkit.containers import IngredientsModule
from astToolkit.transformationTools import makeDictionaryFunctionDef
from hunterHearsPy.theSSOT import settingsPackage
from hunterMakesPy import raiseIfNone
from typing import TYPE_CHECKING
import ast

if TYPE_CHECKING:
	from pathlib import Path

moduleDestination = 'windowingFunctionsTensor'
moduleSource = 'windowingFunctions'

pathFilenameDestination: Path = settingsPackage.pathPackage / (moduleDestination + settingsPackage.fileExtension)
pathFilenameSource: Path = pathFilenameDestination.with_stem(moduleSource)

def make_windowingFunctionsTensor() -> None:
	"""Generate PyTorch tensor windowing functions from NumPy windowing functions."""
	ingredientsModule = IngredientsModule()

	ingredientsModule.imports.addImportFrom_asStr('__future__', 'annotations')

	ingredientsModule.appendPrologue(statement=Make.If(test=Make.Name('TYPE_CHECKING')
									, body=[Make.ImportFrom('torch.types', [Make.alias('Device')])
											, Make.ImportFrom('typing', [Make.alias('Any')])
											, Make.ImportFrom(settingsPackage.identifierPackage + '.theTypes', [Make.alias('callableReturnsNDArray')])
										]))

	ingredientsModule.appendPrologue(statement=Make.FunctionDef(
		'_convertToTensor'
		, Make.arguments(vararg=Make.arg('arguments', annotation=Make.Name('Any'))
			, kwonlyargs=[Make.arg('callableTarget', annotation=Make.Name('callableReturnsNDArray'))
					, Make.arg('device', annotation=Make.BitOr.join([Make.Name('Device'), Make.Constant(None)]))
					, Make.arg('dtype', annotation=Make.BitOr.join([Make.Attribute(Make.Name('torch'), 'dtype'), Make.Constant(None)]))]
			, kw_defaults=[None, Make.Constant(None), Make.Constant(None)]
			, kwarg=Make.arg('keywordArguments', annotation=Make.Name('Any'))
		)
		, body=[Make.Assign([Make.Name('arrayTarget', ast.Store())]
					, value=Make.Call(Make.Name('callableTarget')
						, listParameters=[Make.Starred(value=Make.Name('arguments'))]
						, list_keyword=[Make.keyword(None, value=Make.Name('keywordArguments'))])
				)
			, Make.If(test=Make.Compare(left=Make.Name('device'), ops=[Make.Is()], comparators=[Make.Constant(None)])
				, body=[Make.Assign([Make.Name('device', ast.Store())], value=Make.Call(Make.Attribute(Make.Name('torch'), 'device'), list_keyword=[Make.keyword('device', value=Make.Constant('cpu'))]))]
			)
			, Make.Return(Make.Call(Make.Attribute(Make.Name('torch'), 'tensor'), list_keyword=[
						Make.keyword('data', value=Make.Name('arrayTarget'))
						, Make.keyword('dtype', value=Make.Or.join([Make.Name('dtype'), Make.Attribute(Make.Name('torch'), 'float32')]))
						, Make.keyword('device', value=Make.Name('device'))
					]))
		]
		, returns=Make.Attribute(Make.Name('torch'), 'Tensor')
	))

	dictionaryFunctionDef: dict[str, ast.FunctionDef] = makeDictionaryFunctionDef(parseLogicalPath2astModule('.'.join([settingsPackage.identifierPackage, moduleSource])))

	ImaIndent: str = ' ' * 4
	ImaReturnsSection: str = f"\n{ImaIndent}Returns"
	docstringDevice: str = f"{ImaIndent}device : Device = torch.device(device='cpu')\n{ImaIndent}{ImaIndent}PyTorch device for `Tensor`.\n"
	docstringDtype: str = f"{ImaIndent}dtype : torch.dtype = torch.float32\n{ImaIndent}{ImaIndent}PyTorch data type for `Tensor`.\n"

	for callableIdentifier, astFunctionDef in dictionaryFunctionDef.items():
		if callableIdentifier.startswith('_'):
			continue

		docstring: ast.Expr = Make.Expr(Make.Constant(
			raiseIfNone(ast.get_docstring(astFunctionDef, clean=False), errorMessage="Where's the windowing function docstring?")
							.replace('\t', ImaIndent).replace(ImaReturnsSection, docstringDevice + docstringDtype + ImaReturnsSection)))

		ingredientsModule.imports.addImportFrom_asStr(settingsPackage.identifierPackage, callableIdentifier)

		argumentSpecification: ast.arguments = raiseIfNone(NodeTourist(Be.arguments, Then.extractIt).captureLastMatch(astFunctionDef))
		args: list[ast.expr] = [Make.Name(ast_arg.arg) for ast_arg in [*argumentSpecification.args, *argumentSpecification.kwonlyargs]]

		list_keyword: list[ast.keyword] = [Make.keyword('callableTarget', Make.Name(callableIdentifier))
										, Make.keyword('device', Make.Name('device'))
										, Make.keyword('dtype', Make.Name('dtype'))]
		if argumentSpecification.kwarg:
			list_keyword.append(Make.keyword(None, value=Make.Name(argumentSpecification.kwarg.arg)))

		argumentSpecification.args.append(Make.arg('device', annotation=Make.BitOr.join([Make.Name('Device'), Make.Constant(None)])))
		argumentSpecification.defaults.append(Make.Constant(None))
		argumentSpecification.args.append(Make.arg('dtype', annotation=Make.BitOr.join([Make.Attribute(Make.Name('torch'), 'dtype'), Make.Constant(None)])))
		argumentSpecification.defaults.append(Make.Constant(None))

		ingredientsModule.appendPrologue(statement=Make.FunctionDef(callableIdentifier + 'Tensor'
			, argumentSpecification
			, body=[docstring
				, Make.Return(Make.Call(Make.Name('_convertToTensor'), listParameters=args, list_keyword=list_keyword))]
			, returns=Make.Attribute(Make.Name('torch'), 'Tensor')
		))

	ingredientsModule.imports.addImportFrom_asStr('typing', 'TYPE_CHECKING')
	ingredientsModule.imports.addImport_asStr('torch')

	ingredientsModule.write_astModule(pathFilenameDestination, settingsPackage.identifierPackage)

	docstringModule = '"""Create PyTorch tensor windowing functions."""\n'
	pathFilenameDestination.write_text(docstringModule + pathFilenameDestination.read_text(encoding="utf-8"), encoding='utf-8')

if __name__ == '__main__':
	make_windowingFunctionsTensor()
